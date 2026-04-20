The paper asks us to consider a mapping $\mathcal F$ where $z^k=\mathcal F(z^{k-1}, \theta)$ representing the block of $T$ L-module steps followed by an H-module. The paper somewhat justifies why $\mathcal F$ converges $z$ to a local fixed point $z^*$ in Figure 3. 

$$z^*=\mathcal{F}(z^*, \theta)$$

$$\frac{\partial \mathcal z^*}{\partial \theta}= \frac{\partial \mathcal F}{\partial z^*}\frac{\partial z^*}{\partial \theta} + \frac{\partial \mathcal F}{\partial \theta}$$

$$\text{Let }J_\mathcal F=\frac{\partial \mathcal F}{\partial z^*}$$

$$\implies \frac{\partial \mathcal F}{\partial \theta}=(I-J_\mathcal F)\frac{\partial z^*}{\partial \theta}$$

$$\implies \frac{\partial z^*}{\partial \theta}=(I-J_\mathcal F)^{-1}\frac{\partial \mathcal F}{\partial \theta}$$

With this we can describe the gradient of $\mathcal L$ as

$$g_{\text{true}}=\frac{\partial \mathcal L}{\partial \theta}=\frac{\partial \mathcal L}{\partial z^*}\cdot\frac{\partial z^*}{\partial \theta}=\frac{\partial \mathcal L}{\partial z^*}(I-J_\mathcal F)^{-1}\frac{\partial \mathcal F}{\partial \theta}$$

Paper assumes that $(I-J_\mathcal F)^{-1}\approx I$, yeilding

$$g_{\text{one-step}}=\frac{\partial \mathcal L}{\partial z^*}\cdot\frac{\partial \mathcal F}{\partial \theta}$$

--- 

What we'd like to know if $g_\text{one-step}^\intercal g_\text{true}>0$, i.e. $g_\text{one-step}$ is a descent direction. This is not well-justified in the paper. 


First, we can define the error from the one-step approximation as

$$E := (I - J_\mathcal F)^{-1} - I$$

$$\implies (I - J_\mathcal F)^{-1} = I + E$$

$$\implies g_{\text{true}}
=
\frac{\partial \mathcal L}{\partial z^*}(I + E)\frac{\partial \mathcal F}{\partial \theta}
=
g_{\text{one-step}} + \frac{\partial \mathcal L}{\partial z^*}E\frac{\partial \mathcal F}{\partial \theta} $$

Now expanding out the inner product,

$$
g_{\text{one-step}}^\intercal g_{\text{true}}
=
g_{\text{one-step}}^\intercal \left(g_{\text{one-step}} + \frac{\partial \mathcal L}{\partial z^*}E\frac{\partial \mathcal F}{\partial \theta}\right)
$$


$$
=
g_{\text{one-step}}^\intercal g_{\text{one-step}}
+
g_{\text{one-step}}^\intercal
\left(
\frac{\partial \mathcal L}{\partial z^*}E\frac{\partial \mathcal F}{\partial \theta}
\right)
$$

$$
=
\|g_{\text{one-step}}\|^2
+
g_{\text{one-step}}^\intercal
\left(
\frac{\partial \mathcal L}{\partial z^*}E\frac{\partial \mathcal F}{\partial \theta}
\right)
$$

Therefore, a simplified sufficient condition for descent is

$$\|g_{\text{one-step}}\|^2>\left|g_{\text{one-step}}^\intercal
\left(
\frac{\partial \mathcal L}{\partial z^*}E\frac{\partial \mathcal F}{\partial \theta}
\right)\right| \implies g_{\text{one-step}}^\intercal g_{\text{true}} > 0$$

Using the Cauchy–Schwarz inequality,

$$
\left|
g_{\text{one-step}}^\intercal
\left(
\frac{\partial \mathcal L}{\partial z^*}E\frac{\partial \mathcal F}{\partial \theta}
\right)
\right|
\le
\|g_{\text{one-step}}\|\,
\left\|
\frac{\partial \mathcal L}{\partial z^*}E\frac{\partial \mathcal F}{\partial \theta}
\right\|
$$

So a sufficient condition is

$$
\left\|
\frac{\partial \mathcal L}{\partial z^*}E\frac{\partial \mathcal F}{\partial \theta}
\right\|
<
\|g_{\text{one-step}}\|
$$

Now by Cauchy-Schwartz again, 

$$\left\|
\frac{\partial \mathcal L}{\partial z^*}E\frac{\partial \mathcal F}{\partial \theta}
\right\|\le
\left\|
\frac{\partial \mathcal L}{\partial z^*}
\right\|\left\|E
\right\|\left\|\frac{\partial \mathcal F}{\partial \theta}
\right\|$$

Assuming $\|J_\mathcal F\|<1$, then by the Neumann Series expansion and Cauchy-Schwartz for the last time, we have

$$E=((I-J_\mathcal F)^{-1}-I)=\sum_{k=1}^\infty J_\mathcal F^k$$

$$\implies \|E\|\le\sum_{k=1}^\infty\|J_\mathcal F^k\|=\frac{\|J_\mathcal F\|}{1-\|J_\mathcal F\|}$$

Yeilding our final sufficient condition

$$\left\|
\frac{\partial \mathcal L}{\partial z^*}
\right\|\frac{\|J_\mathcal F\|}{1-\|J_\mathcal F\|} \left\|\frac{\partial \mathcal F}{\partial \theta}
\right\|<\left\|
\frac{\partial \mathcal L}{\partial z^*}\cdot\frac{\partial \mathcal F}{\partial \theta}
\right\|$$

We won't go beyond this analytically, but all of these terms are known during training besides $\|J_\mathcal F\|$. However, if we choose to use the L-2 norm, then we can estimate this by power iteration. 

For a few samples during training, we'll check that this condition holds, which should tell us if the one-step gradient steps described in the paper and re-implemented by our team are justifiable. 