<script>
	import weights from "../lib/weights.json";
	import { classReads, forward, patchesFromRow, softmax } from "../lib/vim.js";
	import { CLASS_COLORS } from "../lib/color.js";

	export let image;
	export let classes = [];

	let decay = -0.7;
	let count = 9;
	const pulses = [0, 1, 0, 0, 0, 0.7, 0, 0, 0];

	function recur(a) {
		let hidden = 0;
		return pulses.map((input) => {
			hidden = Math.exp(a) * hidden + input;
			return hidden;
		});
	}

	function paint(node, values) {
		const draw = (next) => {
			const scale = 4;
			const cell = 8 * scale;
			node.width = cell * 3;
			node.height = cell * 3;
			const ctx = node.getContext("2d");
			for (let y = 0; y < 24; y += 1) {
				for (let x = 0; x < 24; x += 1) {
					ctx.fillStyle = `rgb(${Math.round(image[0][y][x] * 255)}, ${Math.round(image[1][y][x] * 255)}, ${Math.round(image[2][y][x] * 255)})`;
					ctx.fillRect(x * scale, y * scale, scale, scale);
				}
			}
			next.forEach((value, index) => {
				const x = (index % 3) * cell;
				const y = Math.floor(index / 3) * cell;
				ctx.fillStyle = `rgba(109, 40, 217, ${Math.min(0.8, value * 0.9)})`;
				ctx.fillRect(x, y, cell, cell);
				if (value >= 0.08) {
					ctx.fillStyle = "#fff";
					ctx.font = "12px sans-serif";
					ctx.textAlign = "center";
					ctx.fillText(String(Math.round(value * 100)), x + cell / 2, y + cell / 2 + 4);
				}
			});
		};
		draw(values);
		return { update: draw };
	}

	$: decayValue = Number(decay);
	$: stable = recur(decayValue);
	$: blown = Math.max(...stable.map((value) => Math.abs(value)), 1);
	$: kept = Math.exp(decayValue);
	$: base = image ? forward(weights, image) : null;
	$: reads = base
		? classReads(base.blocks[0].forward, weights.blocks[0].forward.A, weights.meta.clsAt)
		: null;
	$: sides = base
		? [
				["Forward", patchesFromRow(base.blocks[0].forward.influence[weights.meta.clsAt], weights.meta.clsAt)],
				["Backward", patchesFromRow(base.blocks[0].backward.influence[weights.meta.clsAt], weights.meta.clsAt)]
			]
		: [];
	$: paths = image
		? ["both", "forward", "backward"].map((keep) => {
				const probs = softmax(forward(weights, image, { keep }).logits);
				const best = probs.indexOf(Math.max(...probs));
				return { keep, probs, best };
			})
		: [];
	$: pairCost = count * count;
	$: scanCost = count;
</script>

<section class="why" data-col="why">
	<h2>Why this math</h2>
	<div class="grid">
		<article>
			<h3>The state is only allowed to fade</h3>
			<p>
				Each step keeps <b>exp(Δ A)</b> of the previous state. A is negative, so that fraction is below 1 and old patches fade. The model stores A as −exp(·), which cannot become positive.
			</p>
			<label>
				A = {decayValue.toFixed(2)} · keep {kept.toFixed(2)}
				<input type="range" min="-1.5" max="0.8" step="0.05" bind:value={decay} />
			</label>
			<div class="pulses" aria-hidden="true">
				{#each pulses as value}
					<i class:on={value > 0}></i>
				{/each}
			</div>
			<div class="trace">
				{#each stable as value}
					<span style:height={`${Math.min(100, (Math.abs(value) / blown) * 100)}%`}></span>
				{/each}
			</div>
			<p class="note">
				{decayValue < 0
					? "On this side a patch is remembered, then forgotten. That is a usable memory."
					: "On this side the state multiplies every step. A few tokens later it overflows, which is why A is forced to stay negative."}
			</p>
		</article>

		<article>
			<h3>Δ is chosen per token</h3>
			<p>
				A small Δ keeps a long memory, so the class token still sees the first patch. A large Δ forgets at once, so it only sees itself. Older scans used one Δ for the whole sequence. Mamba predicts a new Δ from each token, then softplus forces it to stay positive.
			</p>
			{#if reads}
				<div class="maps">
					{#each [["Learned Δ", reads.learned], ["Δ = 0.05", reads.lasting], ["Δ = 2", reads.forgetful]] as [label, read]}
						<figure>
							<canvas use:paint={read.patches}></canvas>
							<figcaption>{label}<br />{(read.self * 100).toFixed(0)}% stays on CLS</figcaption>
						</figure>
					{/each}
				</div>
			{/if}
			<p class="note">Same picture, same block, three step sizes. The learned map is the selective one.</p>
		</article>

		<article>
			<h3>One direction cannot see the whole square</h3>
			<p>
				The forward scan is causal. From the middle, the class token never receives the patches after it, so those cells stay empty. The backward scan is the same rule in the other direction. These shapes often sit on both sides, so the class can survive with one scan. The empty half of the picture cannot.
			</p>
			<div class="maps">
				{#each sides as [label, values]}
					<figure>
						<canvas use:paint={values}></canvas>
						<figcaption>{label}</figcaption>
					</figure>
				{/each}
			</div>
			<div class="paths">
				{#each paths as path}
					<div>
						<strong>{path.keep === "both" ? "Both" : path.keep === "forward" ? "Forward only" : "Backward only"}</strong>
						<em style:color={CLASS_COLORS[path.best]}>{classes[path.best]}</em>
						{#each path.probs as probability, index}
							<div class="mini">
								<span>{classes[index]}</span>
								<div><b style:width={`${probability * 100}%`} style:background={CLASS_COLORS[index]}></b></div>
							</div>
						{/each}
					</div>
				{/each}
			</div>
		</article>

		<article>
			<h3>Attention would score every pair</h3>
			<p>
				A vision transformer writes one attention score for every pair of patches. The scan walks the sequence once and keeps a state whose width does not grow with the image. Nine patches hide the difference. A real Vim image has about 196.
			</p>
			<label>
				{count} patches
				<input type="range" min="9" max="256" step="1" bind:value={count} />
			</label>
			<div class="costs">
				<div>
					<span>Attention {pairCost.toLocaleString()} scores</span>
					<div><b style:width="100%"></b></div>
				</div>
				<div>
					<span>Scan {scanCost.toLocaleString()} steps</span>
					<div><b class="scan" style:width={`${Math.max(2, (scanCost / pairCost) * 100)}%`}></b></div>
				</div>
			</div>
			<p class="note">
				The recurrence is the price of that saving. Information has to be carried, so it needs a decay, a per-token step size, and a second pass in the other direction.
			</p>
		</article>
	</div>
</section>

<style>
	.why {
		padding: 8px 16px 16px;
		background: #f8fafc;
		border-bottom: 1px solid #e2e8f0;
	}
	h2 {
		margin: 4px 0 10px;
		font-size: 15px;
	}
	.grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 10px;
	}
	article {
		background: white;
		border: 1px solid #e2e8f0;
		border-radius: 12px;
		padding: 12px;
	}
	h3 {
		margin: 0 0 6px;
		font-size: 14px;
	}
	p,
	.note,
	label {
		margin: 0 0 8px;
		color: #334155;
		font-size: 13px;
		line-height: 1.45;
	}
	.note {
		color: #64748b;
		margin-bottom: 0;
	}
	label {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-variant-numeric: tabular-nums;
	}
	.pulses {
		display: flex;
		gap: 4px;
		margin-bottom: 6px;
	}
	.pulses i {
		width: 14px;
		height: 14px;
		border-radius: 3px;
		background: #e2e8f0;
	}
	.pulses i.on {
		background: #7c3aed;
	}
	.trace {
		height: 72px;
		display: flex;
		align-items: flex-end;
		gap: 4px;
		margin-bottom: 8px;
	}
	.trace span {
		flex: 1;
		background: #7c3aed;
		border-radius: 3px 3px 0 0;
		min-height: 2px;
	}
	.maps,
	.paths {
		display: flex;
		gap: 8px;
		margin-bottom: 8px;
	}
	figure {
		margin: 0;
	}
	canvas {
		width: 72px;
		height: 72px;
		image-rendering: pixelated;
		border-radius: 6px;
		display: block;
	}
	figcaption,
	.paths strong,
	.paths em {
		display: block;
		font-size: 11px;
		color: #64748b;
		font-style: normal;
		margin-top: 3px;
	}
	.paths em {
		font-weight: 700;
		margin-bottom: 4px;
	}
	.paths > div {
		flex: 1;
		min-width: 0;
	}
	.mini {
		display: grid;
		grid-template-columns: 52px 1fr;
		gap: 4px;
		align-items: center;
		font-size: 10px;
		color: #64748b;
		margin-top: 2px;
	}
	.mini div,
	.costs div div {
		height: 6px;
		background: #f1f5f9;
		border-radius: 99px;
		overflow: hidden;
	}
	.mini b,
	.costs b {
		display: block;
		height: 100%;
		background: #7c3aed;
		border-radius: 99px;
	}
	.costs {
		display: flex;
		flex-direction: column;
		gap: 6px;
		margin-bottom: 8px;
	}
	.costs span {
		font-size: 12px;
		color: #475569;
	}
	.costs b.scan {
		background: #0f766e;
	}
	@media (max-width: 900px) {
		.grid {
			grid-template-columns: 1fr;
		}
	}
</style>
