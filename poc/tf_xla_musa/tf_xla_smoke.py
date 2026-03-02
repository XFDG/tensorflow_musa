#!/usr/bin/env python3
import os
import time

import tensorflow as tf


def env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


USE_XLA = env_flag("USE_XLA", True)
STEPS = int(os.environ.get("STEPS", "50"))
WARMUP = int(os.environ.get("WARMUP", "5"))

print("TF version:", tf.__version__)
print("USE_XLA:", USE_XLA)
print("Physical devices:", tf.config.list_physical_devices())
print("Visible devices:", tf.config.get_visible_devices())


@tf.function(jit_compile=USE_XLA)
def train_step(x, y, w):
    with tf.GradientTape() as tape:
        pred = tf.matmul(x, w)
        loss = tf.reduce_mean((pred - y) ** 2)
    grad = tape.gradient(loss, [w])[0]
    w.assign_sub(0.01 * grad)
    return loss


x = tf.random.normal([4096, 256])
y = tf.random.normal([4096, 1])
w = tf.Variable(tf.random.normal([256, 1]))

for _ in range(WARMUP):
    train_step(x, y, w)

start = time.time()
last_loss = None
for _ in range(STEPS):
    last_loss = train_step(x, y, w)
elapsed = time.time() - start

print("Final loss:", float(last_loss.numpy()))
print("Elapsed sec:", elapsed)
print("Result device:", last_loss.device)
