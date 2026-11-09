export const GROUPS = {
	how: ["classify", "input"],
	architecture: ["embed", "stream", "probs"],
	patches: ["input"],
	embedding: ["embed"],
	position: ["pos"],
	cls: ["input", "embed", "pos"],
	blocks: ["stream", "norm", "x", "delta", "b", "c", "scan", "gate", "out", "more"],
	norm: ["norm"],
	xz: ["x", "gate"],
	dbc: ["delta", "b", "c"],
	directions: ["scan"],
	scan: ["scan"],
	merge: ["scan", "gate"],
	gate: ["gate"],
	residual: ["stream", "out"],
	drop: ["out"],
	logits: ["logits", "probs"],
	probs: ["probs"],
	temperature: ["temperature", "probs"],
	sampling: ["sampling", "probs"],
	readmap: ["readmap", "input"],
	cover: ["readmap", "probs"],
	why: ["why"],
	mech: ["mech"],
	factors: ["factors"],
	stage: ["stage", "input", "probs"]
};

export function hot(focus, column) {
	if (!focus) return true;
	return (GROUPS[focus] || [focus]).includes(column);
}

export const PAGES = [
	{
		id: "what",
		title: "What is Vision Mamba?",
		focus: null,
		html: `<p>Vision Mamba keeps the transformer block shape and replaces self-attention with a <strong>bidirectional selective scan</strong>. This page runs a tiny one: 24×24 images, 3×3 patches, a class token in the middle, and 4 blocks. It classifies four shapes.</p>`
	},
	{
		id: "how",
		title: "Classify an image",
		focus: "how",
		html: `<p>Pick a shape, or draw your own, then press <strong>Classify</strong>. The picture on the right is that forward pass. The colored bars are real activations, not a diagram of a frozen model.</p>`
	},
	{
		id: "architecture",
		title: "Three stages",
		focus: "architecture",
		html: `<p>Same skeleton as the transformer explainer:</p><ol><li><strong>Embedding</strong> turns patches into vectors.</li><li><strong>Vim blocks</strong> mix those vectors.</li><li><strong>Probabilities</strong> come from the class token.</li></ol>`
	},
	{
		id: "patches",
		title: "Patches",
		focus: "patches",
		html: `<p>The image is cut into a 3×3 grid. Each 8×8 patch is one token, read left to right, top to bottom. Flattened, a patch is 192 numbers: channel, then row, then column.</p>`
	},
	{
		id: "embedding",
		title: "Patch embedding",
		focus: "embedding",
		html: `<p>A learned linear layer maps each patch from 192 numbers to 16. Hover a bar to see the width. The class token is not a patch. It is its own learned vector.</p>`
	},
	{
		id: "position",
		title: "Position embedding",
		focus: "position",
		html: `<p>A scan is a loop over a sequence, so order has to be marked. A learned position vector is <strong>added</strong> to each token, including the class token. GPT-2 does the same add. Newer language models often rotate instead.</p>`
	},
	{
		id: "cls",
		title: "Class token in the middle",
		focus: "cls",
		html: `<p>The class token sits at the middle index, not at the end. A forward scan reaches it from the patches before it. A backward scan reaches it from the patches after it. The classifier reads only this token.</p>`
	},
	{
		id: "blocks",
		title: "Repeated Vim blocks",
		focus: "blocks",
		html: `<p>Each block is layer norm, a bidirectional scan, a gate, and a residual add. This model stacks 4. Use the arrows to open the next block. The bars change because each block reads the residual stream the previous block wrote.</p>`
	},
	{
		id: "norm",
		title: "Layer norm",
		focus: "norm",
		html: `<p>Before mixing, each token is rescaled on its own: mean 0, variance 1, then a learned scale and shift. Tokens do not look at each other here.</p>`
	},
	{
		id: "xz",
		title: "x and the gate z",
		focus: "xz",
		html: `<p>The normalized vector is projected to two streams of 32. <strong>x</strong> is what the scan reads, after a short causal convolution and SiLU. <strong>z</strong> waits and later multiplies the scan output. z is the stand-in for the transformer's MLP.</p>`
	},
	{
		id: "dbc",
		title: "Δ, B, and C",
		focus: "dbc",
		html: `<p>These three are the selective parameters, the role Q, K, and V play in attention.</p><ul><li><strong>Δ</strong> is a positive step size. Large Δ forgets the state faster.</li><li><strong>B</strong> writes this token into the state.</li><li><strong>C</strong> reads the state back out.</li></ul><p>Each direction has its own Δ, B, and C.</p>`
	},
	{
		id: "directions",
		title: "Forward and backward",
		focus: "directions",
		html: `<p>Attention runs many heads in parallel. Vision Mamba runs two scans. Forward walks the patches from the start. Backward walks them from the end, with its own convolution, Δ, B, C, and decay A. Switch direction the way you would switch a head.</p>`
	},
	{
		id: "scan",
		title: "Selective scan",
		focus: "scan",
		html: `<p>The state update is</p><pre>hₜ = exp(Δₜ A) hₜ₋₁ + Δₜ Bₜ xₜ\nyₜ = Cₜ · hₜ + D xₜ</pre><p>A is a fixed negative diagonal, so exp(Δ A) is a decay between 0 and 1. Each row of the matrix is the share of the state readout that came from each visible token. Forward only sees tokens at or before itself. Backward only sees tokens at or after itself. Hatched cells are that mask.</p>`
	},
	{
		id: "merge",
		title: "Add the two directions",
		focus: "merge",
		html: `<p>The forward output and the backward output are added, token by token, before the gate. After that add, every patch has had a chance to influence the class token.</p>`
	},
	{
		id: "gate",
		title: "Gate",
		focus: "gate",
		html: `<p>The merged scan is multiplied by SiLU(z). SiLU lets a large positive number through and squashes a negative one toward zero. In a transformer this slot is the MLP, applied to each token alone. The scan already did the mixing.</p>`
	},
	{
		id: "residual",
		title: "Residual stream",
		focus: "residual",
		html: `<p>The block adds its output back onto its input. The next block reads that sum. The thin bars on the left of a later block are this stream, not the original embedding.</p>`
	},
	{
		id: "drop",
		title: "Drop path",
		focus: "drop",
		html: `<p>Training can randomly skip the residual add so the block becomes the identity. That is drop path, the stand-in for dropout. It is off while classifying, so the add always happens. The transformer explainer shows dropout the same way: present in the diagram, inactive at inference.</p>`
	},
	{
		id: "logits",
		title: "Logits",
		focus: "logits",
		html: `<p>After the last block, layer norm runs on every token. Only the class token is multiplied by the classifier matrix. Four scores, one per shape. The other tokens stop here.</p>`
	},
	{
		id: "probs",
		title: "Probabilities",
		focus: "probs",
		html: `<p>Softmax turns the four scores into probabilities that sum to 1. The outlined class is the one just sampled. The longest bar is the mode. They match when the distribution is sharp.</p>`
	},
	{
		id: "temperature",
		title: "Temperature",
		focus: "temperature",
		html: `<p>Temperature divides the logits before softmax. Low temperature makes the largest score take nearly all the mass. High temperature flattens the four classes. The scan does not rerun. Only the readout changes.</p>`
	},
	{
		id: "sampling",
		title: "Sampling",
		focus: "sampling",
		html: `<p><strong>Top-k</strong> keeps the k largest classes. <strong>Top-p</strong> keeps the smallest set whose mass reaches p. The kept mass is renormalized, then <strong>Draw class</strong> samples one. Classify draws again after a fresh forward pass.</p>`
	},
	{
		id: "readmap",
		title: "Where a token looks",
		focus: "readmap",
		html: `<p>The purple on the image is one row of the scan matrix, drawn back onto the 3×3 patches. <strong>Both</strong> adds the forward and backward reads. Forward can only light patches at or before the chosen token. Backward can only light patches at or after it. The four small frames are the class token at each block.</p>`
	},
	{
		id: "cover",
		title: "Cover a patch",
		focus: "cover",
		html: `<p>Switch to <strong>Hide</strong> and click a patch. That patch is replaced with gray and the model runs again. The number under each class is how its probability moved compared with the full picture. Lifting your pen also reruns the drawing.</p>`
	},
	{
		id: "why",
		title: "Why this math",
		focus: "why",
		html: `<p>Four comparisons, each against the mechanism that was not chosen. Drag <strong>A</strong> across zero to see the state explode. Compare a learned <strong>Δ</strong> with a step size that is too small or too large. Switch off one scan direction. Then grow the patch count and watch attention's pair scores pull away from the scan's single pass.</p>`
	},
	{
		id: "mech",
		title: "The step, the token, the gate",
		focus: "mech",
		html: `<p>The keep fraction <strong>exp(Δ A)</strong> is taken from this block and applied to a unit pulse. No decay never lets go of the first pulse. Forgetting half drops it immediately. The three grids show a class token at the start, the middle, and the end: a scan cannot jump, so the middle is the short carry. The last card reruns the picture with the gate removed and with the convolution removed.</p>`
	},
	{
		id: "stage",
		title: "Watch it move",
		focus: "stage",
		html: `<p>The pen classifies while you draw. <strong>Mix</strong> blends two shapes and the scores follow the slider. <strong>Play depth</strong> keeps only the first blocks, one frame at a time, so you can see the class change as blocks are added. The gold square on the picture is the patch the scan is on while it plays.</p>`
	},
	{
		id: "factors",
		title: "Change one piece",
		focus: "factors",
		html: `<p>Each card reruns the trained model with one substitution. The gate buttons swap SiLU for ReLU, GELU, sigmoid, tanh, or no gate, and the strips are the gate gradient: ReLU is exactly zero on negative channels, so those channels stop learning. The scan activation is a different slot, the function after the convolution. The third card flips the sign of <strong>A</strong> and enlarges <strong>Δ</strong>. Negative A decays, so a bigger step only forgets. Positive A grows on every step, the state overflows, and the class changes. The table is the structural edits that move the four shapes: the residual, the depth, and the convolution.</p>`
	}
];

export const POPOVERS = {
	patches: {
		title: "Patches",
		html: `<p>Image 24×24×3, patch 8. Nine tokens, row-major. Inside a token the values run channel, then row, then column.</p>`
	},
	embedding: {
		title: "Patch embedding",
		html: `<p>Linear map <span class="mono">192 → 16</span>, plus a bias. The class token skips this map. It is a learned vector of length 16, inserted at index 4.</p>`
	},
	position: {
		title: "Position embedding",
		html: `<p>Table of shape <span class="mono">10 × 16</span>, one row per token including the class token. Added to the embedding.</p>`
	},
	norm: {
		title: "Layer norm",
		html: `<p>For one token x,</p><pre>y = (x − mean) / sqrt(var + ε) · γ + β</pre><p>γ and β are learned. ε is 1e-5. Each token is normalized alone.</p>`
	},
	x: {
		title: "Input projection",
		html: `<p>Linear map <span class="mono">16 → 64</span>, split in half. The first 32 numbers are x. They go through a depthwise causal convolution of width 3, then SiLU, then the scan. The other 32 are z.</p>`
	},
	delta: {
		title: "Δ, the step size",
		html: `<p>From the convolved x: a bias-free map <span class="mono">32 → 4</span>, then <span class="mono">4 → 32</span>, then softplus so every entry is positive. Δ multiplies both the decay and the write.</p><pre>Ā = exp(Δ A)    B̄ = Δ B</pre>`
	},
	b: {
		title: "B, the write",
		html: `<p>Four numbers per token, from the same projection as Δ. B decides what this token writes into the state. There is one B for the forward scan and another for the backward scan.</p>`
	},
	c: {
		title: "C, the read",
		html: `<p>Four numbers per token. The readout is the dot product of C with the state, plus a skip D·x. C is the value-side of the scan: it chooses which part of the memory leaves.</p>`
	},
	scan: {
		title: "Selective scan",
		html: `<pre>hₜ = exp(Δₜ A) ⊙ hₜ₋₁ + (Δₜ Bₜ) xₜ\nyₜ = Cₜ · hₜ + D xₜ</pre><p>State width is 4, hidden width is 32, so each of the 32 channels has its own decay. The matrix is the state path C·h only, normalized to sum to 1 like an attention row. The local skip D x is not in the cells. Hatched cells are exactly zero: that direction's mask.</p>`
	},
	gate: {
		title: "SiLU gate",
		html: `<pre>SiLU(z) = z · sigmoid(z)\ny = (y_fwd + y_bwd) ⊙ SiLU(z)</pre><p>Then a linear map takes 32 numbers back to 16.</p>`
	},
	residual: {
		title: "Residual",
		html: `<pre>stream ← stream + out</pre><p>The next block, and the final layer norm, read this sum.</p>`
	},
	drop: {
		title: "Drop path",
		html: `<p>During training a block can be skipped with some probability, which leaves the residual stream unchanged. At classify time the probability is 0. Nothing is dropped.</p>`
	},
	logits: {
		title: "Classifier",
		html: `<p>Final layer norm, then a linear map <span class="mono">16 → 4</span> applied only to the class token. The four numbers are the logits.</p>`
	},
	softmax: {
		title: "Softmax",
		html: `<pre>pᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)</pre><p>T is the temperature. The bars are these probabilities after top-k or top-p has zeroed the rejected classes and the rest has been renormalized.</p>`
	},
	temperature: {
		title: "Temperature",
		html: `<p>T &lt; 1 sharpens. T &gt; 1 flattens. The Vim blocks are not rerun.</p>`
	},
	sampling: {
		title: "Top-k and top-p",
		html: `<p>Top-k keeps the k largest probabilities. Top-p sorts the probabilities and keeps the shortest prefix whose sum is at least p. Draw class samples from what remains.</p>`
	}
};
