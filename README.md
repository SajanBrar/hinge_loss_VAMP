# VAMP experiments for hinge and hinge squared losses

## Analysis of the hinge loss using VAMP

In this section we consider the Hinge-loss formulation of soft-margin SVM and formulate the VAMP algorithm and its State Evolution for this case.

Let $\mathbf{A} \in \mathbb{R}^{m \times n}$ be the data matrix, $\mathbf{y} \in \{-1, +1\}^m$ the output label vector, and $\mathbf{x}$ be the vector we want to learn.

Define the hinge loss:

$$
L(\mathbf{Ax}, \mathbf{y}) = \sum_i [1 - y_i (\mathbf{Ax})_i]_+
$$

Let $\text{svd}(\mathbf{A}) = \mathbf{USV^\top}$, $\boldsymbol{\beta} := \mathbf{V^\top x}$, and $\boldsymbol{\nu} := \mathbf{SV^\top}\mathbf{x}$. Denote by $\mathbf{S}^\dagger$ the pseudo-inverse of $\mathbf{S}$. Define $\alpha := n/m$.

Now the SVM objective, $\min_{\mathbf{x} \in \mathbb{R}^n} L(\mathbf{Ax}, \mathbf{y}) + \frac{\lambda}{2} \|\mathbf{x}\|^2_2$ can be written as:

$$
\begin{aligned}
    &\min_{\boldsymbol{\nu} \in \mathbb{R}^m} L(\mathbf{U}\boldsymbol{\nu}, \mathbf{y}) + \lambda \rho(\boldsymbol{\nu}) \\
    \text{where}& \quad \rho(\boldsymbol{\nu}) = \frac{1}{2} \|\mathbf{S}^\dagger \boldsymbol{\nu}\|^2_2 + \delta_\mathcal{S}(\boldsymbol{\nu}) \\
    \text{and }& \quad \mathcal{S} := \{\mathbf{z} \in \mathbb{R}^m \mid z_i = 0 \ \forall i > \text{rank}(\mathbf{A})\}
\end{aligned}
$$

The dirac-delta, $\delta$, is $\infty$ outside $\mathcal{S}$ and $0$ otherwise. This can be performed as a projection onto the set $\mathcal{S}$. Define the mask vector $\mathbf{m}$ with $m_i = 0 \ \forall i > \text{rank}(\mathbf{A})$ and $1$ otherwise. Then $\text{Proj}_{\mathcal{S}}(\mathbf{q}) = \mathbf{q} \odot \mathbf{m}$.

Define the following using proximal operators:

$$
H(\mathbf{q}, \eta) := \text{prox}_{L/\eta}(\mathbf{q}), \qquad G(\mathbf{r}, \gamma) := \text{prox}_{\rho/\gamma}(\mathbf{r})
$$

Define divergence as $\langle \mathbf{u} \rangle := \frac{1}{N} \sum_{n=1}^N u_n$, and $\mathbf{f}'(\mathbf{r}, \gamma) = \mathrm{diag}(\nabla_\mathbf{r} \mathbf{f}(\mathbf{r}, \gamma))$.

Also consider the squared version of the Hinge Loss:

$$
L^2(\mathbf{A}\mathbf{x}, \mathbf{y}) = \sum_i \left( [ 1 - y_i (\mathbf{A} \mathbf{x})_i ]_+ \right)^2
$$

$$
\begin{aligned}
    H(q_i, \eta) := \text{prox}_{L^2/\eta}(q_i) &= \arg \min_{z_i} \left[ \frac{1}{\eta} \left( \max \left( 0, 1 - y_i z_i \right)\right)^2 + \frac{1}{2}(z_i - q_i)^2 \right] 
    = \begin{cases} 
        q_i & \text{if } y_i q_i \ge 1 \\ 
        \frac{\eta q_i + 2y_i}{\eta + 2} & \text{if } y_i q_i < 1 
    \end{cases} \\
    G(r_i, \gamma) := \text{prox}_{\rho\lambda/\gamma}(r_i) &= \arg\min_{\mathbf{z}} \left[ \frac{\lambda}{2\gamma} \|\mathbf{S}^\dagger \mathbf{z}\|^2_2 + \frac{1}{2} \|\mathbf{z} - \mathbf{r}\|^2_2 \right]_i 
    = \mathbf{s}_i^2 \left( \mathbf{s}_i^2 + (\lambda/\gamma) \right)^{-1} r_i m_i
\end{aligned}
$$

And:

$$
H'(q_i, \eta) = \begin{cases}
    1 & \text{if } y_i q_i > 1 \\
    \frac{\eta}{\eta + 2} & \text{if } y_i q_i < 1
\end{cases}, \qquad 
G'(r_i, \gamma) = \mathbf{s}_i^2 \left( \mathbf{s}_i^2 + (\lambda/\gamma) \right)^{-1}
$$

### VAMP Algorithm

Here $\mathbf{z}, \mathbf{q}, \mathbf{r}, \boldsymbol{\theta}, \mathbf{m} \in \mathbb{R}^m$ and $\delta, \nu, \gamma, \eta \in \mathbb{R}$:

$$
\begin{aligned}
    \mathbf{z}_t &= H(\mathbf{q}_t, \delta_t) \\
    \nu_t &= \frac{\delta_t}{\langle H'(\mathbf{q}_t, \delta_t)\rangle} \\
    \gamma_t &= \nu_t - \delta_t \\
    \mathbf{r}_t &= \frac{\mathbf{U}^\top (\nu_t \mathbf{z}_t - \delta_t\mathbf{q}_t)}{\gamma_t} \\
    \boldsymbol{\theta}_t &= G(\mathbf{r}_t, \gamma_t) \\
    \eta_t &= \frac{\gamma_t}{\langle G'(\mathbf{r}_t, \gamma_t) \rangle} \\
    \delta_{t+1} &= \eta_t - \gamma_t \\
    \mathbf{q}_{t+1} &= \frac{\mathbf{U}(\eta_t \boldsymbol{\theta}_t - \gamma_t \mathbf{r}_t)}{\delta_{t+1}}
\end{aligned}
$$

### State Evolution

Remember $PL(k)$ convergence means that for a random sequence of vectors $\{x^{(n)}\}$, $n \in \mathbb{N}$, $x^{(n)} \in \mathbb{R}^n$, $x \stackrel{PL(k)}{\to} X$ if:

$$
\lim_{n \to \infty} \frac{1}{n} \sum_{i=0}^n \psi(x_i^{(n)}) \stackrel{a.s.}{=} \mathbb{E}[\psi(X)] \quad \text{for every } \psi \in PL(k, 1).
$$

Assume $\lim_{n \to \infty} \{ (\mathbf{z}^*, \mathbf{s}, \boldsymbol{\theta}^*) \} \stackrel{PL(2)}{=} (Z^*, S, \Theta^*)$. Under certain additional assumptions, in the LSL the following holds almost surely for all $t \geq 0$:

$$
\begin{aligned}
    \lim_{n \to \infty} \{(\boldsymbol{\theta}^*, \mathbf{r}_{t} - \boldsymbol{\theta}^*)\} &\stackrel{PL(2)}{=} (\Theta^*, R_{t}) \\
    \lim_{n \to \infty} \{(\mathbf{z}^*, \mathbf{q}_{t} - \mathbf{z}^*)\} &\stackrel{PL(2)}{=} (Z^*, Q_{t})
\end{aligned}
$$

We have:

$$
\begin{aligned}
    R_t \sim \mathcal{N}(0, \tau_t^2), &\quad Q_t \sim \mathcal{N}(0, \mu_t^2), \\ 
    Z_t = H(Q_t + Z^*, \bar{\delta}_t), &\quad \Theta_t = G(R_t + \Theta^*, \bar{\gamma}_t).
\end{aligned}
$$

Additionally, almost surely:

$$
\lim_{n \to \infty} (\nu_t, \gamma_t, \delta_t, \eta_t ) = (\bar{\nu}_t, \bar{\gamma}_t, \bar{\delta}_t, \bar{\eta}_t)
$$

Finally, the State Evolution equations for the Hinge Loss are as follows ($\alpha := n/m$). Initialize $\mu_0$, $\check{\delta}_0$, and $Z^*, X^*$ follow the empirical distribution, with $Q_t \sim \mathcal{N}(0, \mu_t^2)$ and $R_t \sim \mathcal{N}(0, \tau_t^2)$.

The variances are then tracked by the State Evolution:

$$
\begin{aligned}
    Z_t &= H(Q_t + Z^*, \bar{\delta}_t) \\
    \bar{\nu}_t &= \frac{\bar{\delta}_t}{\mathbb{E}[H'(Q_t + Z^*, \bar{\delta}_t)]} \\ 
    \bar{\gamma}_t &= \bar{\nu}_t - \bar{\delta}_t \\
    \tau_t^2 &= \text{Var}\left( \frac{\bar{\nu}_t(Z_t - Z^*) - \bar{\delta}_t Q_t}{\bar{\gamma}_t} \right) \\
    \Theta_t &= G(R_t + \Theta^*, \bar{\gamma}_t) \\
    \bar{\eta}_t &= \frac{\bar{\gamma}_t}{\mathbb{E}[G'(R_t + \Theta^*, \bar{\gamma}_t)]} \\
    \bar{\delta}_{t+1} &= \bar{\eta}_t - \bar{\gamma}_t \\ 
    \mu_{t+1}^2 &= \text{Var}\left( \frac{\bar{\eta}_t(\Theta_t - \Theta^*) - \bar{\gamma}_t R_t}{\bar{\delta}_{t+1}}\right)
\end{aligned}
$$

Let $(\Theta^*, R_t) \sim \mathcal{N}\left(\mathbf{0}, \begin{bmatrix} \sigma_1^2 & \tilde{\sigma}_t \\ \tilde{\sigma}_t & \tau_t^2 \end{bmatrix} \right)$ and $(Z^*, Q_t) \sim \mathcal{N}\left(\mathbf{0}, \begin{bmatrix} \sigma_2^2 & \check{\sigma}_t \\ \check{\sigma}_t & \mu_t^2 \end{bmatrix} \right)$.

It can be shown that (where $\mathrm{T}, \mathrm{B}, \mathrm{V}$ are defined below):

$$
\begin{aligned}
    \mathbb{E}[G'(R_t + \Theta^*, \bar{\gamma}_t)] &= \mathbb{E}\left[ \frac{S^2}{S^2 + \lambda / \bar{\gamma}_t} \right] = \alpha\left(1 - \mathrm{T}\right) \\
    \mu_{t+1}^2 &= \frac{1}{\bar{\delta}_{t+1}^2} \left[ \tau_{\perp}^2 \left( \bar{\eta}_t^2 \alpha(1 - 2\mathrm{T} + \mathrm{B}) - \bar{\gamma}_t^2 \right) + \sigma_1^2 \left( c^2\bar{\delta}_{t+1}^2 - 2c\bar{\delta}_{t+1}\bar{\eta}_t(c+1)u(1-\mathrm{T}) + \bar{\eta}_t^2(c+1)^2 u^2 \mathrm{V} \right) \right]
\end{aligned}
$$

where $c = \tilde{\sigma}_t / \sigma_1^2$, $\tau_{\perp}^2 = \tau_t^2 - c \tilde{\sigma}_t$, $u = \lambda / \bar{\gamma}_t$, and $\mathrm{T}, \mathrm{B}, \mathrm{V}$ are the standard Marchenko-Pastur integral functions evaluated at $(\alpha, u)$.

Let $X \in \mathbb{R}^{m \times n}$ have i.i.d. random entries. Eigenvalues of $Y = \frac{1}{n}XX^\top$ follow the Marchenko-Pastur Law (with $\alpha := n/m$). The following quantities show up in our analysis:

$$
\begin{aligned}
    \mathrm{T} &:= \mathrm{T}\left(\alpha, \frac{\lambda}{\bar{\gamma}_t}\right) = \mathbb{E}\left[ \frac{\frac{\lambda}{\bar{\gamma}_t}}{\frac{\lambda}{\bar{\gamma}_t} + S^2} \right] \\
    \mathrm{B} &:= \mathrm{B}\left(\alpha, \frac{\lambda}{\bar{\gamma}_t}\right) = \mathbb{E}\left[ \left( \frac{\frac{\lambda}{\bar{\gamma}_t}}{\frac{\lambda}{\bar{\gamma}_t} + S^2} \right)^2 \right] \\
    \mathrm{V} &:= \mathrm{V}\left(\alpha, \frac{\lambda}{\bar{\gamma}_t}\right) = \mathbb{E}\left[ \frac{S^2}{\left(\frac{\lambda}{\bar{\gamma}_t} + S^2\right)^2} \right]
\end{aligned}
$$

Additionally, it holds:

$$
\mathrm{B}(\alpha, u) = \mathrm{T}(\alpha, u) - u \frac{\partial}{\partial u} \mathrm{T}(\alpha, u), \qquad
\mathrm{V}(\alpha, u) = \frac{\partial}{\partial u} \mathrm{T}(\alpha, u)
$$

$$
\mathrm{T}(\alpha, u) = \frac{-1 + \alpha - u + \sqrt{(1 + u - \alpha)^2 + 4\alpha u}}{2\alpha}
$$
