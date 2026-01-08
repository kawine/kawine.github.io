---
layout: post
title: "To Scale Economics, Scale Language"
date: 2026-01-06
author: Kawin Ethayarajh
excerpt: Economists are finally becoming scaling-pilled. But to circumvent the Lucas critique, we need to scale the right thing.
---

Economists are becoming scaling-pilled.[^0]

A recent post by Alex Imas and Arpit Gupta asks whether Transformers can learn economic relationships.[^1] They train a Transformer on data simulated from a New Keynesian model and find that it handily beats traditional VAR approaches by an order of magnitude. Their conclusion: structure should be "something which can be learned, rather than assumed."

This is the right instinct. For decades, the bitter lesson[^2] has played out across AI: general methods that leverage computation beat approaches built on human-encoded knowledge, every time. Chess engines that searched deeper beat engines stuffed with grandmaster heuristics. Language models that scaled on web text beat models with hand-crafted grammars and ontologies. The pattern is so consistent that it should update our priors on any field where we currently rely on stylized structural assumptions—economics included.

But here's my concern: the conversation on scaling economic models has largely focused on _time series_. This seems to be a hangover from how models are currently built, and such models would scale poorly. To successfully scale in economics—to the point where we can circumvent the Lucas critique—we need to scale models that _reason in language_.

## The Lucas Critique, Revisited

Imas and Gupta have a great summary of Robert Lucas's famous 1976 argument:[^3]

> ... economists had been interpreting correlational relationships from historical data as structural, meaning they were invariant to policy changes. But they weren’t. The economic agents which generated the data may change their behavior in reaction to changes in policy, which—as the Phillips curve example showed—can shift the observed relationship between variables.

The Phillips curve proposed an inverse relationship between inflation and unemployment. Policymakers thought they could choose a point on the downward-sloping menu, trading off low unemployment for higher inflation (or vice versa) using fiscal/monetary tools. However, this relationship broke down in the 1970s amidst supply shocks and changing expectations of inflation.

The standard response to Lucas's critique has been to build "structural" models—models that explicitly represent agents' preferences, beliefs, and constraints. The idea is that these deep parameters are invariant to policy, so if you get them right, your model survives regime changes.

But in practice, getting them right is very hard. Structural models require strong assumptions that are often wrong, and the models—because they need to be human-legible—are much simpler than the true underlying structure. And even when they're approximately right, they miss the rich and nuanced way in which agents actually form expectations: by browsing the Internet, watching the news, and talking to each other. If economic decisions are downstream of reasoning in language, our economic models should reason in language too.[^5]

## Why Time-Series Is Not All You Need

The Imas–Gupta result is a useful one, but it’s best read as a proof of capacity: a sufficiently large Transformer can, in principle, learn NK-like structure from data without being handed the structure upfront. In their setup, that’s possible because the model can be trained on arbitrarily many samples from a known simulation.

In practice, we have to learn from real economic data. And real economic data has three properties that make time-series-only scaling a dead end.

### 1. Data Scarcity

Large models are data-hungry, but economic regimes change faster than data accumulates. We get one Great Recession, one COVID shock, one period of post-pandemic inflation. By the time you have enough data to learn the structure of a regime, you're in a new regime.

Consider COVID-era inflation. In early 2020, a purely time-series model would have seen decades of stable, low inflation. Nothing in the historical CPI series would suggest that inflation was about to spike to 9%. The structural break happened because of something outside the time series: a worldwide pandemic that disrupted supply chains, followed by unprecedented fiscal and monetary stimulus.

If regimes change faster than you can learn them, you're always fitting stale structure.

Imas–Gupta circumvent this by sampling as needed from their NK simulation. But we can't conjure arbitrary samples of real-world regime changes. What we can do is leverage a different (and massively larger) source of information: the vast corpus of human reasoning about the world, encoded in language.

### 2. Computational Inefficiency

Time series models waste compute on redundant information. Consider weekly inflation readings over a stable period. Each data point is highly correlated with its neighbors; the marginal information per observation is low. Yet an attention-based model must attend over all of them, incurring $O(n^2)$ cost in sequence length $n$.

This mirrors a known issue in language modeling. In principle, you could train on raw bytes instead of tokens, preserving more information. In practice, nobody does this because the improvement in performance is marginal (if there is any improvement to begin with) while the computational cost is prohibitive—you'd be spending attention on individual characters when words or subwords suffice.

The same logic applies to economic time series. Instead of feeding a model 52 weekly inflation readings, you could describe the same information as: "inflation was stable at 2.1% throughout 2019." Is this lossy? Yes. But you can now use all the leftover space in your context window to cram other important information, like unemployment rates, consumer confidence, and more.

Not only is this more efficient, it's a much *more natural* representation of how economic information is actually communicated and acted upon. Policymakers don't stare at raw time series; they read summary reports, memos, and news articles. A language-based approach lets the model operate at the appropriate level of abstraction.

### 3. Modality Constraints

A time series model can only learn from time series. But economic agents form expectations from everything: news articles, earnings calls, policy speeches, charts, even memes. A model that ingests only one modality is blind to signals that agents actually respond to.

This blindness makes the model maximally vulnerable to the Lucas critique. It can only detect regime changes *as they manifest in the series*, which means it's always lagging. By the time inflation ticks up, agents have already read the news about supply chain disruptions and adjusted their expectations.

A reasoning model that processes language (and images, and tables, and summaries of time series) can infer structural pressure before it prints in the data. It can read a central bank speech and infer hawkishness from emphasis and framing, or parse a pandemic briefing and reason about downstream effects on supply and demand. Even before LLMs took off, institutional investors used traditional NLP models to infer latent beliefs from text—this is a scaled-up version of that idea.

## The Case for Reasoning Models

What we need is a model that can *reason* about economic structure, not just fit it. This is where large reasoning models—models that can deliberate at inference time over evidence before answering—start to matter.

Consider the COVID inflation example again. In February 2020, a reasoning model could read:

1. News reports about a novel coronavirus spreading in Wuhan
2. Government statements about possible lockdowns
3. Analysis pieces about supply chain dependencies on China

From this, it could reason: "A global pandemic seems likely. Lockdowns will disrupt production. Fiscal stimulus is probable. Supply-constrained economies with excess demand will see inflation." This is not a prediction from simply fitting $y_t$ on $y_{t-1}$, but a prediction from understanding how the world works.

### Circumventing the Lucas Critique

A reasoning model has a shot at anticipating how agents change their behavior in response to regime changes if the model is conditioned on the proposed intervention and can simulate how agents would update.

This is qualitatively different from fitting reduced-form relationships. Instead of learning "when $x$ goes up, $y$ goes down", the model learns *why* $y$ goes down when $x$ goes up—the mechanism, the agent reasoning, and the (dis)equilibrium logic. And when the mechanism changes, it can update accordingly.

Formally, let $r_t \in \mathcal{R}$ denote the economic regime at time $t$, and let $\mathcal{L}_t$ denote the set of language signals (news, policy statements, etc.) available at time $t$. A reasoning model learns:

$$p(r_{t+1} | \mathcal{L}_t) \quad \text{and} \quad p(y_{t+1} | r_{t+1}, \mathcal{L}_t)$$

That is, it infers the likely regime from language and predicts outcomes conditional on that regime. A pure time series model, by contrast, can only learn $p(y_{t+1} \| y_{1:t})$, which collapses regimes into a single unconditional distribution. When regimes shift, the time series model's predictions are biased; the reasoning model can adapt, because it has access to the signals that precipitate regime change.

### Making Everything In-Distribution

In language modeling, the scaling-believer's response to out-of-distribution concerns was simple: _make everything in-distribution_. If you have enough data, interpolation begins to look a lot like extrapolation, even if you don't have true out-of-domain generalization.

The same logic applies here. Train a reasoning model on the vast corpus of economic analysis—academic papers, Fed minutes, analyst reports, news coverage, historical commentary, and so on—because every economic event has been discussed, analyzed, and debated in language. 

For a model that has seen all this, the set of counterfactual regimes that are both realistic and truly out-of-distribution should shrink dramatically. Time series are sparse; regime changes are rare events in the data. But every crisis is analyzed exhaustively after the fact, and often anticipated beforehand, so a model that learns from this corpus can generalize in ways that a time series model cannot.

## It Won't Be Easy

**Computational cost.** Reasoning models generate many tokens before producing an answer. At inference time, this can be expensive—potentially prohibitive for real-time forecasting applications. The scale at which you can get a generally useful model is likely beyond what you can create in academia.

**Auditability.** Structural econometric models are transparent: you can inspect the assumptions, trace the logic, debate the parameter choices. A reasoning model's chain-of-thought is less legible. This is analogous to the contrast between linguistics (where we want to _understand_ language) and language modeling (where we want to _predict_ language). Embracing scaling requires accepting discomfort with not understanding.

**Reproducibility.** Running the same prompt twice may give you different reasoning chains. For policy applications that demand consistency and accountability, this is a problem.

These are real concerns, but they're addressable. Compute costs are falling, interpretability research is advancing, and models are becoming more steerable. People had many of the same concerns about scaling LLMs, and slowly but surely, we've been able to address most of them.

## Conclusion

The economists getting scaling-pilled are on the right track: structure should be learned, not assumed, and general methods that leverage computation will win.

But it's not enough to scale; we need to scale the right thing. The best option we have is throw reasoning models at the vast corpus of human economic thought. That's where the structure lives, and that's how we'll build models that don't just fit the past, but anticipate the future.

To scale economics, scale language.

---

[^0]: Scaling-pilled is slang for belief in the [scaling hypothesis](https://gwern.net/scaling-hypothesis).

[^1]: Imas, A. & Gupta, A. (2026). [Can a Transformer Learn Economic Relationships?](https://aleximas.substack.com/p/can-a-transformer-learn-economic)

[^2]: Sutton, R. (2019). [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html). The argument: general methods that leverage computation outperform methods built on human knowledge, consistently across 70 years of AI research.

[^3]: Lucas, R. (1976). Econometric Policy Evaluation: A Critique. Carnegie-Rochester Conference Series on Public Policy.

[^5]: Leland Bybee has made a similar point across his papers, e.g., [The Ghost in the Machine: Generating Beliefs with Large Language Models](https://lelandbybee.com/files/LLM.pdf). 