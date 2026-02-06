
***[Deep reinforcement learning from human preferences, Christian et. al. 2023](https://arxiv.org/abs/1706.03741)***

(I have already read this paper previously, so notes are pretty raw)


## Setting

- We have observations $\mathcal{O}$ and actions $\mathcal{A}$
- We have trajectories $\sigma = ((o_0, a_0), \ldots, (o_{k-1}), a_{k-1}) \in (\mathcal{O} \times \mathcal{A})^k$
- We have human preference $\sigma^1 \succ \sigma^2$ on trajectories
- Goal: produce trajectories preferred by the human.
  
  In particular, if the preference relation can be described by a *reward function*, then the optimal policy should be close to the optimal policy for the reward function.

This is particularly useful in cases where it is **hard to explicitly specify a reward function but easy to verify output**. In the paper they use it to train a toy agent to backflip.

## Methodology

The authors approach this as a *reward learning* problem:
- We train a reward function $\hat{r}$ and a policy $\pi$
- Full loop:
	- Sample trajectories according to $\pi$
	- Optimize $\pi$ to maximize $\hat{r}$ on these trajectories
	- Sample human feedback on these trajectories
	- Optimize $\hat{r}$ according to this feedback
	- Repeat

**Details:**
$\pi$ can be optimized according to $\hat{r}$ using any algorithm. Perhaps [REINFORCE](https://arxiv.org/abs/2010.11364) is a simple one since it is an RL algorithm that only optimizes the policy.

For optimizing $\hat{r}$, roughly the idea here is to use something similar to the **Elo system** in chess. In the Elo system, if someone with rating $R_1$ plays someone with rating $R_1$, then we assume the ratings are such that the first player wins with probability $$\frac{\exp(R_1)}{\exp(R_1) + \exp(R_2)}$$
and then $R_1$ and $R_2$ are continuously tuned to match the actual sampled data (using cross-entropy loss).

Similarly, we assume that the probability that $\sigma^1 \succ \sigma^2$ is equal to

$$
\frac{\exp(J(\sigma^1))}{\exp(J(\sigma^1)) + \exp(J(\sigma^2))}
$$
where $$J(\sigma^1) := \sum_i \hat{r}(o_i, a_i)$$
gives the total reward. Then we optimize $\hat{r}$ using cross-entropy loss.

**Execution Tweaks:**
- They fit $k$ reward functions and average the results
- $\frac{1}{e}$ of the data is held out to use as a validation set for each predictor
- softmax is modified under the assumption there's a $10\%$ chance the human responds at random (smoothing out some of the tails).
- They had some technique for selecting which preferences to collect human feedback on. I'll read this later.

