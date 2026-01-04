# code for "What Affects Effective Depth of Large Language Models?"

We investigate the factors that affect the effective depth of LLMs, including model scales, training strategies and task difficulties.

We follow "Do Language Models Use Their Depth Efficiently?" by Csordás et al. (2025) and use the following metrics, the code is adapted from [their repository](https://github.com/robertcsordas/llm_effective_depth):
1. [Relative norm contribution & cosine similarity](./plot_relative_contributions.py)
2. [The effects of the layer on future computations](./plot_layer_effects.py)
3. [Logit lens](./plot_logit_lens.py)
4. [Residual erasure experiment](./plot_residual_erasure.py)
5. [Integrated gradients](./plot_integrated_gradients.py)

