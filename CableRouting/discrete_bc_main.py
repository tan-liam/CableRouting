import os
import time
from copy import deepcopy
import uuid
import re
from functools import partial

import numpy as np
import pprint

import jax
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
import optax

import absl.app
import absl.flags

import faiss

from .jax_utils import (
    JaxRNG,
    next_rng,
    named_tree_map,
    discrete_encode,
    discrete_decode,
)
from .model import ResNetPolicy
from .data import (
    partition_batch_train_test,
    subsample_batch,
    preprocess_robot_dataset,
    augment_batch,
    get_data_augmentation,
)
from .utils import (
    define_flags_with_default,
    set_random_seed,
    print_flags,
    get_user_flags,
    prefix_metrics,
    WandBLogger,
    average_metrics,
)


FLAGS_DEF = define_flags_with_default(
    seed=42,
    dataset_path="",
    dataset_image_keys="side_image",
    image_augmentation="none",
    codebook_size=64,
    kmeans_n_iters=1000,
    kmeans_n_redo=20,
    clip_action=0.99,
    train_ratio=0.9,
    batch_size=128,
    total_steps=10000,
    lr=1e-4,
    lr_warmup_steps=0,
    weight_decay=0.05,
    clip_gradient=1e9,
    log_freq=50,
    eval_freq=200,
    eval_batches=20,
    save_model=False,
    policy=ResNetPolicy.get_default_config(),
    logger=WandBLogger.get_default_config(),
)
FLAGS = absl.flags.FLAGS


def main(argv):
    assert FLAGS.dataset_path != ""
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logger, variant=variant)
    set_random_seed(FLAGS.seed)

    image_keys = FLAGS.dataset_image_keys.split(":")
    dataset = np.load(FLAGS.dataset_path, allow_pickle=True).item()
    dataset = preprocess_robot_dataset(dataset, FLAGS.clip_action)

    kmeans = faiss.Kmeans(
        dataset["action"].shape[-1],
        FLAGS.codebook_size,
        niter=FLAGS.kmeans_n_iters,
        nredo=FLAGS.kmeans_n_redo,
        verbose=True,
        gpu=True,
    )
    kmeans.train(dataset["action"])
    codebook = kmeans.centroids
    discretization_error, _ = kmeans.assign(dataset["action"])
    del kmeans
    wandb_logger.log({"discretization_error": np.mean(discretization_error)})

    train_dataset, test_dataset = partition_batch_train_test(dataset, FLAGS.train_ratio)

    policy = ResNetPolicy(output_dim=FLAGS.codebook_size, config_updates=FLAGS.policy,)

    params = policy.init(
        state=train_dataset["robot_state"][:5, ...],
        images=[dataset[key][:5, ...] for key in image_keys],
        rngs=next_rng(policy.rng_keys()),
    )

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=FLAGS.lr,
        warmup_steps=FLAGS.lr_warmup_steps,
        decay_steps=FLAGS.total_steps,
        end_value=0.0,
    )

    def weight_decay_mask(params):
        def decay(name, _):
            for rule in ResNetPolicy.get_weight_decay_exclusions():
                if re.search(rule, name) is not None:
                    return False
            return True

        return named_tree_map(decay, params, sep="/")

    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.clip_gradient),
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=FLAGS.weight_decay,
            mask=weight_decay_mask,
        ),
    )
    train_state = TrainState.create(params=params, tx=optimizer, apply_fn=None)
    codebook = jax.device_put(codebook)

    @partial(jax.jit, donate_argnums=1)
    def train_step(rng, train_state, state, action, images):
        rng_generator = JaxRNG(rng)
        action_labels = discrete_encode(codebook, action)

        def loss_fn(params):
            logits = policy.apply(
                params, state, images, rngs=rng_generator(policy.rng_keys())
            )
            loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, action_labels)
            )
            predicted_action = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean((predicted_action == action_labels).astype(jnp.float32))
            reconstructed = discrete_decode(codebook, predicted_action)
            mse = jnp.mean(jnp.sum(jnp.square(reconstructed - action), axis=-1))
            return loss, (accuracy, mse)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (accuracy, mse)), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            mse=mse,
            learning_rate=learning_rate(train_state.step),
        )
        return rng_generator(), train_state, metrics

    @jax.jit
    def eval_step(rng, train_state, state, action, images):
        rng_generator = JaxRNG(rng)
        action_labels = discrete_encode(codebook, action)
        logits = policy.apply(
            train_state.params, state, images, rngs=rng_generator(policy.rng_keys())
        )
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, action_labels)
        )
        predicted_action = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean((predicted_action == action_labels).astype(jnp.float32))
        reconstructed = discrete_decode(codebook, predicted_action)
        mse = jnp.mean(jnp.sum(jnp.square(reconstructed - action), axis=-1))
        metrics = dict(eval_loss=loss, eval_accuracy=accuracy, eval_mse=mse,)
        return rng_generator(), metrics

    augmentation = get_data_augmentation(FLAGS.image_augmentation)
    rng = next_rng()

    best_loss, best_accuracy, best_mse = float("inf"), -0.1, float("inf")
    best_loss_model, best_accuracy_model, best_mse_model = None, None, None

    for step in range(FLAGS.total_steps):
        batch = subsample_batch(train_dataset, FLAGS.batch_size)
        batch = augment_batch(augmentation, batch)
        rng, train_state, metrics = train_step(
            rng,
            train_state,
            batch["robot_state"],
            batch["action"],
            [batch[key] for key in image_keys],
        )
        metrics["step"] = step

        if step % FLAGS.log_freq == 0:
            wandb_logger.log(metrics)
            pprint.pprint(metrics)

        if step % FLAGS.eval_freq == 0:
            eval_metrics = []
            for _ in range(FLAGS.eval_batches):
                batch = subsample_batch(test_dataset, FLAGS.batch_size)
                rng, metrics = eval_step(
                    rng,
                    train_state,
                    batch["robot_state"],
                    batch["action"],
                    [batch[key] for key in image_keys],
                )
                eval_metrics.append(metrics)
            eval_metrics = average_metrics(jax.device_get(eval_metrics))
            eval_metrics["step"] = step

            if eval_metrics["eval_loss"] < best_loss:
                best_loss = eval_metrics["eval_loss"]
                best_loss_model = jax.device_get(train_state)

            if eval_metrics["eval_accuracy"] > best_accuracy:
                best_accuracy = eval_metrics["eval_accuracy"]
                best_accuracy_model = jax.device_get(train_state)

            if eval_metrics["eval_mse"] < best_mse:
                best_mse = eval_metrics["eval_mse"]
                best_mse_model = jax.device_get(train_state)

            eval_metrics["best_loss"] = best_loss
            eval_metrics["best_accuracy"] = best_accuracy
            eval_metrics["best_mse"] = best_mse
            wandb_logger.log(eval_metrics)
            pprint.pprint(eval_metrics)

            if FLAGS.save_model:
                save_data = {
                    "variant": variant,
                    "step": step,
                    "codebook": jax.device_get(codebook),
                    "train_state": jax.device_get(train_state),
                    "best_loss_model": best_loss_model,
                    "best_accuracy_model": best_accuracy_model,
                    "best_mse_model": best_mse_model,
                }
                wandb_logger.save_pickle(save_data, f"model.pkl")

    if FLAGS.save_model:
        save_data = {
            "variant": variant,
            "step": step,
            "codebook": jax.device_get(codebook),
            "train_state": jax.device_get(train_state),
            "best_loss_model": best_loss_model,
            "best_accuracy_model": best_accuracy_model,
            "best_mse_model": best_mse_model,
        }
        wandb_logger.save_pickle(save_data, f"model.pkl")


if __name__ == "__main__":
    absl.app.run(main)
