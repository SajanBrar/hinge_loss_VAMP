# VAMP experiments for hinge and hinge squared losses. 

\section{Analysis of the hinge loss using VAMP}
In this section we consider the Hinge-loss formulation of soft-margin SVM and formulate the VAMP algorithm and it's State Evolution for this case. \\
Let $\mathbf{A} \in \mathbb{R}^{m \times n}$ be the data matrix, $\mathbf{y} \in \{-1, +1\}^m$ the output label vector, and $\mathbf{x}$ be the vector we want to learn.

Define the hinge loss:
\begin{equation}\label{eq:hinge}
        L(\mathbf{Ax}, \mathbf{y}) = \sum_i [1 - y_i (\mathbf{Ax})_i]_+ 
\end{equation} 
Let $\text{svd}(\mathbf{A}) = \mathbf{USV^\top}$, $\boldsymbol{\beta} := \mathbf{V^\top x}$, and $\boldsymbol{\nu} := \mathbf{SV^\top}\mathbf{x} $. Denote by $\mathbf{S}^\dagger$ the pseudo-inverse of $\mathbf{S}$. Define $\alpha := n/m$.\\
Now the SVM objective, $\min_{\mathbf{x} \in \mathbb{R}^n} L(\mathbf{Ax}, \mathbf{y}) + \frac{\lambda}{2} \|\mathbf{x}\|^2_2$ can be written as: 
\begin{align}
    &\min_{\boldsymbol{\nu} \in \mathbb{R}^m} L(\mathbf{U}\boldsymbol{\nu}, \mathbf{y}) + \lambda \rho(\boldsymbol{\nu})  \\
    \text{where}& \quad \rho(\boldsymbol{\nu}) = \frac{1}{2} \|\mathbf{S}^\dagger \boldsymbol{\nu}\|^2_2 + \delta_\mathcal{S}(\boldsymbol{\nu}) \\
    \text{and }& \quad \mathcal{S} := \{\mathbf{z} \in \mathbb{R}^m |z_i = 0 \text{ }\forall i> \text{rank}(\mathbf{A})\}
\end{align}
The dirac-delta, $\delta$ is $\infty$ outside $\mathcal{S}$ and $0$ otherwise. This can be performed as a projection onto the set $\mathcal{S}$. Define the mask vector $\mathbf{m}$ with $m_i = 0 \text{ } \forall i> \text{rank}(\mathbf{A}) \text{ and } 1 \text{ otherwise}$. Then $Proj_{\mathcal{S}} (\mathbf{q}) = \mathbf{q} \odot \mathbf{m} $.
Define the following using proximal operators:
\begin{align*}
    H(\mathbf{q}, \eta) := prox_{L/\eta}(\mathbf{q}) \quad \quad \quad \quad
    G(\mathbf{r}, \gamma) := prox_{\rho/\gamma}(\mathbf{r}) 
\end{align*}
Define divergence as $\langle \mathbf{u} \rangle := \frac{1}{N} \sum_{n=1}^N u_n$, and $\mathbf{f}'(\mathbf{r}, \gamma) = \mathrm{diag}(\nabla_\mathbf{r} \mathbf{f}(\mathbf{r}, \gamma))$.
Also consider the squared version of the Hinge Loss,
\begin{equation}
    L^2(\mathbf{A}\mathbf{x}, \mathbf{y}) = \sum_i \left( [ 1 - y_i (\mathbf{A} \mathbf{x})_i ]_+ \right)^2
\end{equation}
\begin{align*}
    H(q_i, \eta) := prox_{L^2/\eta}(q_i) &= \arg \min_{z_i} \bigg[ \frac{1}{\eta} \left( \max \left( 0, 1 - y_i z_i \right)\right)^2 + \frac{1}{2}(z_i - q_i)^2 \bigg] \nonumber 
    = \begin{cases} 
q_i & \text{if } y_i q_i \ge 1 \\ 
\frac{\eta q_i + 2y_i}{\eta + 2} & \text{if } y_i q_i < 1 
\end{cases} \\
G({r}_i, \gamma) := prox_{\rho\lambda/\gamma}({r}_i) &= \arg\min_\mathbf{z} \bigg[ \frac{\lambda}{2\gamma} \|\mathbf{S}^\dagger \mathbf{z}\|^2_2 + \frac{1}{2} \|\mathbf{z}- \mathbf{r}\|^2_2 \bigg]_i\nonumber 
% \bigg[ \mathbf{S}^2 \left( \mathbf{S}^2 + (\lambda/\gamma)\mathbf{I} \right)^{-1} \mathbf{r} \odot \mathbf{m} \nonumber \bigg]_i 
 = \mathbf{s}_i^2 \left( \mathbf{s}_i^2 + (\lambda/\gamma) \right)^{-1} r_i m_i
\end{align*}
And, 
\begin{align*}
    H'(q_i, \eta) = \begin{cases}
1 & \text{if } y_i q_i > 1 \\
\frac{\eta}{\eta + 2} & \text{if } y_i q_i < 1
\end{cases}  \quad \quad \quad \quad 
G'({r}_i, \gamma) = \mathbf{s}_i^2 \left( \mathbf{s}_i^2 + (\lambda/\gamma) \right)^{-1}
\end{align*}
VAMP (here $\mathbf{z}$, $\mathbf{q}$, $\mathbf{r}$, $\boldsymbol{\theta}$, $\mathbf{m}$ $\in \mathbb{R}^m$ and $\delta$, $\nu$, $\gamma$, $\eta$ $\in \mathbb{R}$.):
\begin{subequations}
    \begin{align}
    \mathbf{z}_t &= H(\mathbf{q}_t ,\delta_t) \\
    \nu_t &= \frac{\delta_t}{\langle H'(\mathbf{q}_t ,\delta_t)\rangle} \\
    \gamma_t &= \nu_t - \delta_t \\
    \mathbf{r}_t &= \frac{\mathbf{U}^\top (\nu_t \mathbf{z}_t - \delta_t\mathbf{q}_t)}{\gamma_t} \\
    \boldsymbol{\theta}_t &= G(\mathbf{r}_t, \gamma_t)\\
    \eta_t &= \frac{\gamma_t}{\langle G'(\mathbf{r}_t, \gamma_t) \rangle} \\
    \delta_{t+1} &= \eta_t - \gamma_t \\
    \mathbf{q}_{t+1} &= \frac{\mathbf{U}(\eta_t \boldsymbol{\theta}_t - \gamma_t \mathbf{r}_t)}{\delta_{t+1}}
\end{align}
\end{subequations}

Remember $PL(k)$ convergence means that for a random sequence of vectors 
${x^{(n)}}$, ${n \in \mathbb{N}}$, $x^{(n)} \in \mathbb{R}^n$,
$x \stackrel{PL(k)}{\to} X$ if,
\begin{align*}
\lim_{n \to \infty} \frac{1}{n} \sum_{i=0}^n \psi(x_i^{(n)}) \stackrel{a.s.}{=} \mathbb{E}[\psi(X)] \quad \text{for every } \psi \in PL(k, 1).
\end{align*}
Assume $\lim_{n \to \infty} \{ (\mathbf{z}^*, \mathbf{s}, \boldsymbol{\theta}^*) \} \stackrel{PL(2)}{=} (Z^*, S, \Theta^*)$. \\
    Under certain more assumptions, in the LSL the following holds almost surely for all $t\geq0$:
\begin{subequations} 
        \begin{align}
        \lim_{n \to \infty} \{(\boldsymbol{\theta}^*_{}, \mathbf{r}_{t} - \boldsymbol{\theta}^*)\} &\stackrel{PL(2)}{=} (\Theta^*, R_{t}) \\
        \lim_{n \to \infty} \{(\mathbf{z}^*_{}, \mathbf{q}_{t} - \mathbf{z}^*)\} &\stackrel{PL(2)}{=} (Z^*, Q_{t})
\end{align}
\end{subequations}
We have: 
\begin{align*}
    R_t  \sim \mathcal{N}(0, \tau_t^2), \quad &Q_t \sim \mathcal{N}(0, \mu_t^2), \\ Z_t = H(Q_t + Z^*, \bar{\delta}_t), \quad &{\Theta_t} = G(R_t + \Theta^*, \bar{\gamma}_t). 
\end{align*}
Additionally, almost surely,
\begin{align} \label{eq:scalarsconverge}
\lim_{n \to \infty} (\nu_t, \gamma_t, \delta_t, \eta_t ) = (\bar{\nu}_t, \bar{\gamma}_t, \bar{\delta}_t, \bar{\eta}_t)    
\end{align}
The undefined variables in the above equations are the variables in the state evolution equations below. 
Finally, the State Evolution equations for the Hinge Loss are as follows ($\alpha := n/m$.):
Initialize $\mu_0$, $\check{\delta}_0$, and $Z^*$, $X^*$ follow the empirical distribution. $Q_t \sim \mathcal{N}(0, \mu_t^2)$, $R_t \sim \mathcal{N}(0, \tau_t^2)$. \\

The variances are then tracked by the State Evolution:
\begin{subequations}
\begin{align}
    Z_t &= H(Q_t + Z^*, \bar{\delta}_t) \\
    \bar{\nu}_t &= \frac{\bar{\delta}_t}{\mathbb{E}[H'(Q_t + Z^*, \bar{\delta}_t)]} \\ \bar{\gamma}_t &= \bar{\nu}_t - \bar{\delta}_t \\
    \tau_t^2 &= Var\left( \frac{\bar{\nu}_t(Z_t - Z^*) - \bar{\delta}_tQ_t}{\bar{\gamma}_t} \right) \\
    {\Theta}_t &= G(R_t + \Theta^*, \bar{\gamma}_t)\\
    \bar{\eta}_t &= \frac{\bar{\gamma}_t}{\mathbb{E}[G'(R_t + \Theta^*, \bar{\gamma}_t)]} \\
    \bar{\delta}_{t+1} &= \bar{\eta}_t - \bar{\gamma}_t \\ 
    \mu_{t+1}^2 &= Var\left( \frac{\bar{\eta}_t({\Theta}_t - \Theta^*) - \bar{\gamma}_t R_t}{\bar{\delta}_{t+1}}\right)
\end{align}
\end{subequations}

Let $(\Theta^*, R_t) \sim \mathcal{N}\left(\mathbf{0}, \begin{bmatrix} \sigma_1^2 & \tilde{\sigma}_t \\ \tilde{\sigma}_t & \tau_t^2 \end{bmatrix} \right)$, $(Z^*, Q_t) \sim \mathcal{N}\left(\mathbf{0}, \begin{bmatrix} \sigma_2^2 & \check{\sigma}_t \\ \check{\sigma}_t & \mu_t^2 \end{bmatrix} \right)$.\\

It can be shown that (T, B, V will be defined below) :
\begin{subequations}
\begin{align}
    \mathbb{E}[G'(R_t + \Theta^*, \bar{\gamma}_t)] &= \mathbb{E}\bigg[ \frac{S^2}{S^2 + \lambda / \bar{\gamma}_t} \bigg] = \alpha\left(1 - \mathrm{T}\right) \\
        \mu_{t+1}^2 &= \frac{1}{\bar{\delta}_{t+1}^2} \left[ \tau_{\perp}^2 \left( \bar{\eta}_t^2 \alpha(1 - 2\mathrm{T} + \mathrm{B}) - \bar{\gamma}_t^2 \right) + \sigma_1^2 \left( c^2\bar{\delta}_{t+1}^2 - 2c\bar{\delta}_{t+1}\bar{\eta}_t(c+1)u(1-\mathrm{T}) + \bar{\eta}_t^2(c+1)^2 u^2 \mathrm{V} \right) \right]
\end{align}
\end{subequations}
where $c = \tilde{\sigma}_t / \sigma_1^2$, $\tau_{\perp}^2 = \tau_t^2 - c \tilde{\sigma}_t$, $u = \lambda / \bar{\gamma}_t$, and $\mathrm{T}, \mathrm{B}, \mathrm{V}$ are the standard Marchenko-Pastur integral functions evaluated at $(\alpha, u)$.

Let $X \in \mathbb{R}^{m \times n}$ have $iid$ random entries. Eigenvalues of $Y = \frac{1}{n}XX^\top$ follow the Marchenko-Pastur Law (with $\alpha := n/m$). The following quantities show up in our analysis:\\
\begin{subequations}
\begin{align}
\mathrm{T} &:= \mathrm{T}\left(\alpha, \frac{\lambda}{\bar{\gamma}_t}\right) = \mathbb{E}\Bigg[ \frac{\frac{\lambda}{\bar{\gamma}_t}}{\frac{\lambda}{\bar{\gamma}_t}+ S^2} \Bigg] \\
\mathrm{B} &:= \mathrm{B}\left(\alpha, \frac{\lambda}{\bar{\gamma}_t}\right) = \mathbb{E}\Bigg[ \left( \frac{\frac{\lambda}{\bar{\gamma}_t}}{\frac{\lambda}{\bar{\gamma}_t} + S^2 } \right)^2 \Bigg] \\
\mathrm{V} &:= \mathrm{V}\left(\alpha, \frac{\lambda}{\bar{\gamma}_t}\right) = \mathbb{E}\Bigg[ \frac{S^2}{(\frac{\lambda}{\bar{\gamma}_t} + S^2)^2}\Bigg]
\end{align}
\end{subequations}
Additionally it holds:
\begin{align*}
    \mathrm{B}(\alpha, u) = \mathrm{T}(\alpha, u) - u \frac{\partial}{\partial u} \mathrm{T}(\alpha, u), \quad
    \mathrm{V}(\alpha, u) = \frac{\partial}{\partial u} \mathrm{T}(\alpha, u) \\
        \mathrm{T}(\alpha, u) = \frac{-1+\alpha-u+\sqrt{(1+u-\alpha)^2+4\alpha u}}{2\alpha} \quad \quad 
\end{align*}
