<script>
	export let open = false;
</script>

{#if open}
	<div class="backdrop" on:click={() => (open = false)} role="presentation"></div>
	<section class="article" role="dialog" aria-label="About this model">
		<header>
			<h2>How to read this model</h2>
			<button type="button" on:click={() => (open = false)}>Close</button>
		</header>
		<p>
			Vision Mamba (Vim) is a vision transformer whose mixer is a selective state-space scan
			instead of attention. This page is the same walk the transformer explainer gives GPT-2:
			your input, the embedding, one block opened in place, the blocks after it, then the
			probabilities, temperature, and sampling.
		</p>
		<h3>What is the same</h3>
		<ul>
			<li>Patches stand in for tokens. A learned vector is added for position.</li>
			<li>Blocks repeat. Each one starts with layer norm and ends with a residual add.</li>
			<li>The gate is per token. It does the job of the MLP: a nonlinearity that does not mix tokens.</li>
			<li>The class scores go through temperature, top-k or top-p, then a sample.</li>
		</ul>
		<h3>What replaces attention</h3>
		<pre>hₜ = exp(Δₜ A) ⊙ hₜ₋₁ + (Δₜ Bₜ) xₜ
yₜ = Cₜ · hₜ + D xₜ
y = (y_forward + y_backward) ⊙ SiLU(z)</pre>
		<p>
			Δ, B, and C are produced from the token, so the scan is selective. A is a fixed negative
			diagonal: the decay. Forward can see the token and the ones before it. Backward can see
			the token and the ones after it. The hatched cells in the matrix are that mask. The two
			outputs are added, then gated.
		</p>
		<h3>Why this math, and not the neighbor</h3>
		<p>
			exp(Δ A) is a fraction of the old state only when A is negative, so A is stored as −exp(·).
			A single Δ would forget every token at the same speed; a per-token Δ, kept positive by
			softplus, can drop the background and hold the shape. One causal scan cannot see the
			patches on the other side of the class token, so there are two. Attention would instead
			score every pair of patches. That cost grows with the square of the image, which is the
			reason for the recurrence.
		</p>
		<h3>Why the class token is in the middle</h3>
		<p>
			A one-way scan would need the whole image to pass before a token at the end could see the
			first patch. Putting the class token in the middle lets each direction reach it from the
			patches on its own side. The read map paints that row of the scan back onto the image.
			Covering a patch reruns the same network with that patch blanked, which is how you see
			whether the class actually depended on it.
		</p>
		<h3>What was scaled down</h3>
		<p>
			A published Vim-tiny model is 192-wide and 24 blocks deep, trained on ImageNet. This one is
			16-wide, 4 blocks deep, and trained on circles, squares, stripes, and diagonals. The width
			is the part that changed. The dataflow is the part the picture is about. Drop path is drawn
			and left off, the same way dropout is inactive when GPT-2 is generating.
		</p>
	</section>
{/if}

<style>
	.backdrop {
		position: fixed;
		inset: 0;
		background: rgb(15 23 42 / 35%);
		z-index: 30;
	}
	.article {
		position: fixed;
		top: 0;
		right: 0;
		height: 100%;
		width: min(440px, 100%);
		background: #fff;
		z-index: 31;
		overflow: auto;
		padding: 22px 22px 40px;
		box-shadow: -12px 0 40px rgb(15 23 42 / 12%);
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
	}
	h2 {
		margin: 0;
		font-size: 20px;
	}
	h3 {
		margin: 18px 0 6px;
		font-size: 15px;
	}
	p,
	li {
		color: #334155;
		line-height: 1.5;
		font-size: 14px;
	}
	button {
		border: 1px solid #cbd5e1;
		background: #fff;
		border-radius: 8px;
		padding: 6px 10px;
		cursor: pointer;
		font: inherit;
	}
	pre {
		background: #f8fafc;
		border: 1px solid #e2e8f0;
		border-radius: 8px;
		padding: 10px 12px;
		font-size: 12.5px;
		line-height: 1.5;
		overflow-x: auto;
	}
</style>
