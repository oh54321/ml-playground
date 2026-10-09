# Live PPO training graphs on Runpod

The trainer writes TensorBoard event files into each run's
`artifacts/ppo_2048/<run>/tensorboard` directory when launched with
`--tensorboard`. It records training losses, PPO diagnostics, gradient metrics,
exploration, learning rate, iteration time, and evaluation scores. Most PPO
metrics are averaged across the update epochs in each outer iteration;
`approximate_kl` and `clip_fraction` use the final update epoch's values. The
`ppo/update_epochs` series records how many update epochs completed in each
outer iteration, including iterations that stopped early at the KL threshold.
The positive `entropy` series is logged without the redundant `negative_entropy`
series. PyTorch's
`SummaryWriter` produces the event files TensorBoard reads; see the
[PyTorch TensorBoard documentation](https://docs.pytorch.org/docs/2.14/tensorboard.html).

Evaluation logs `evaluation/win_rate`, the fraction of games whose terminal
board reached 2048 or a higher tile. Each run also saves
`evaluation_win_rate.png`. Older checkpoints do not contain this metric, and
their win rate cannot be recovered from scores.

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

The training agent starts TensorBoard detached on the Pod, bound to
`127.0.0.1:6006`, with `artifacts/ppo_2048` as its log directory. It writes the
server log to `/workspace/tensorboard.log` and verifies that the HTTP endpoint
responds. If TensorBoard is already running, it reuses it; the server discovers
new event files automatically.

On your computer, run the simple SSH tunnel command with the Pod IP and port
from the Pod's **SSH over exposed TCP** details in Runpod's Connect tab. This
forwards local port 6007 to Pod port 6006 and discards the host key instead of
adding it to `~/.ssh/known_hosts`:

```sh
ssh -N -L 6007:127.0.0.1:6006 -p <SSH_PORT> -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o LogLevel=ERROR root@<POD_PUBLIC_IP>
```

Because this skips SSH host-key verification, use it only for this disposable
Runpod tunnel. Keep the terminal open and visit <http://localhost:6007>. Press
**Ctrl+C** to close only the tunnel; training, TensorBoard, and the Pod
continue running. If SSH needs a specific identity, add `-i <SSH_KEY_PATH>` to
the command. Full SSH requires a public IP and SSH access enabled on the Pod;
Runpod's
[SSH guide](https://docs.runpod.io/pods/configuration/use-ssh) covers setup.
The tunnel keeps TensorBoard off Runpod's public HTTP proxy, where exposed HTTP
services are reachable over the internet
([port exposure guide](https://docs.runpod.io/pods/configuration/expose-ports)).

## Stop a run and keep its checkpoint

Press **Ctrl+C** in the training terminal for a graceful stop. The trainer
updates `latest.pt` and the score PNGs after every completed evaluation; it
updates `best.pt` only when the mean score improves. A stop between evaluations
keeps the most recent completed evaluation checkpoint, so a few later training
iterations may need to be repeated. After stopping the run, stop the Pod in the
Runpod console to end GPU charges.

The full diagnostic PNG set is also written when the trainer exits. The
evaluation score PNGs and TensorBoard charts update while it is running.
