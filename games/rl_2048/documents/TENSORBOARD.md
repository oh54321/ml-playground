# Live PPO training graphs on Runpod

The trainer writes TensorBoard event files into each run's
`artifacts/ppo_2048/<run>/tensorboard` directory when launched with
`--tensorboard`. It records training losses, PPO diagnostics, gradient metrics,
exploration, learning rate, iteration time, and evaluation scores. Most PPO
metrics are averaged across the update epochs in each outer iteration;
`approximate_kl` and `clip_fraction` use the final update epoch's values. The
positive `entropy` series is logged without the redundant `negative_entropy`
series. PyTorch's
`SummaryWriter` produces the event files TensorBoard reads; see the
[PyTorch TensorBoard documentation](https://docs.pytorch.org/docs/2.14/tensorboard.html).

## Start a monitored run

From the repository root on the Pod, install the optional dependency and start
a short run that evaluates often:

```sh
python -m pip install tensorboard
python -m games.rl_2048.train_2048 \
  --tensorboard \
  --outer-iterations 20 \
  --evaluation-interval 2 \
  --evaluation-games 25
```

The evaluation overrides are useful for an early stop decision. They leave the
saved defaults alone. Increase `--evaluation-games` for steadier score estimates
once a run looks promising.

## Open the charts from your computer

In a second Pod terminal, from the repository root, start TensorBoard listening
only inside the Pod:

```sh
tensorboard --logdir artifacts/ppo_2048 --host 127.0.0.1 --port 6006 --reload_interval 5
```

For access from your computer, use the Pod's **full SSH (public IP)** connection
details from Runpod's Connect tab. Full SSH needs a public IP and an SSH daemon;
Runpod's official PyTorch template is preconfigured, while a custom template
needs TCP port 22 exposed. Their [SSH guide](https://docs.runpod.io/pods/configuration/use-ssh)
has the current setup details.

On your computer, open another terminal and create a local SSH tunnel. Replace
the placeholders with the `root@IP`, port, and key from the Pod's **SSH over
exposed TCP** command:

```sh
ssh -N -L 6006:127.0.0.1:6006 -p <SSH_PORT> -i <SSH_KEY_PATH> root@<POD_PUBLIC_IP>
```

Keep that terminal open, then open <http://localhost:6006> in your browser. This
keeps TensorBoard off Runpod's public HTTP proxy. Runpod notes that exposed HTTP
services are reachable over the internet, so the tunnel avoids publishing an
unauthenticated dashboard ([port exposure guide](https://docs.runpod.io/pods/configuration/expose-ports)).

## Stop a run and keep its checkpoint

Press **Ctrl+C** in the training terminal for a graceful stop. The trainer
updates `latest.pt` and the score PNGs after every completed evaluation; it
updates `best.pt` only when the mean score improves. A stop between evaluations
keeps the most recent completed evaluation checkpoint, so a few later training
iterations may need to be repeated. After stopping the run, stop the Pod in the
Runpod console to end GPU charges.

The full diagnostic PNG set is also written when the trainer exits. The
evaluation score PNGs and TensorBoard charts update while it is running.
