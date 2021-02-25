# Bayesian-Deep-Reinforcement-Learning
In many online sequential decision-making problems, such as contextual bandits [4] and
reinforcement learning [17], an agent learns to take a sequence of actions to maximize its
expected cumulative reward, while repeatedly interacting with an unknown environment.
Moreover, since in such problems the agent’s actions affect both its rewards and its ob-
servations, it faces the well-known exploration-exploitation trade-off. Consequently, the
exploration strategy is crucial for a learning algorithm:


Under-exploration will typically yield a sub-optimal strategy, while over-exploration tends
to incur a significant exploration cost. Various exploration strategies have been proposed,
including epsilon-greedy (EG), Boltzmann exploration [16] upper-confidence-bound (UCB)
[3] type exploration, and Thompson sampling (TS). Among them, TS [18], which is also
known as posterior sampling or probability matching, is a widely used exploration strategy
with good practical performance [5] and theoretical guarantees [13] [1].



Because the exact posterior is intractable, evaluating these approaches is hard. Further-
more, these methods are rarely compared on benchmarks that measure the quality of their
estimates of uncertainty for downstream tasks. To address this challenge, we develop a
benchmark for exploration using deep neural networks.
In this project, we investigate how the posterior approximations affect the performance
of Thompson Sampling from an empirical standpoint. We test the performance in four
different typical recommendation Datasets.
