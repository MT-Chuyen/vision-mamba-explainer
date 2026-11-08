<script>
	import weights from "../lib/weights.json";
	import { forward, recurrenceTrace, softmax } from "../lib/vim.js";
	import { CLASS_COLORS } from "../lib/color.js";

	export let image;
	export let classes = [];

	const pulse = [0, 1, 0, 0, 0, 0.7, 0, 0, 0];

	function roll(keep) {
		let hidden = 0;
		return pulse.map((input) => {
			hidden = keep * hidden + input;
			return hidden;
		});
	}

	function hops(place) {
		return Array.from({ length: 9 }, (_, patch) => {
			if (place === "start") return patch + 1;
			if (place === "end") return 9 - patch;
			const index = patch < 4 ? patch : patch + 1;
			return Math.abs(index - 4);
		});
	}

	function belief(options) {
		const probs = softmax(forward(weights, image, options).logits);
		const best = probs.indexOf(Math.max(...probs));
		return { probs, best };
	}

	$: trace = image ? recurrenceTrace(forward(weights, image).blocks[0].forward, weights.blocks[0].forward.A) : [];
	$: keep = trace.length ? trace[weights.meta.clsAt].decay : 0.96;
	$: memories = [
		["This block", roll(keep)],
		["No decay", roll(1)],
		["Forget half", roll(0.5)]
	];
	$: peak = Math.max(...memories.flatMap((entry) => entry[1]), 1);
	$: placements = [
		["Class token first", hops("start")],
		["Class token in the middle", hops("middle")],
		["Class token last", hops("end")]
	];
	$: mechanisms = image
		? [
				["SiLU gate and conv", belief({})],
				["No gate", belief({ gate: "none" })],
				["No convolution", belief({ conv: false })]
			]
		: [];
</script>

<section class="mech" data-col="mech">
	<h2>The step, the class token, the gate</h2>
	<div class="grid">
		<article>
			<h3>What one update keeps</h3>
			<p>
				At the class token this block keeps <b>{(keep * 100).toFixed(0)}%</b> of the state, because
				<code>exp(Δ A) = {keep.toFixed(2)}</code>. The bars use that real fraction on a unit pulse, so the shape is easy to see. The network's own writes are smaller. The shape is the point.
			</p>
			<div class="pulse" aria-hidden="true">
				{#each pulse as value}
					<i class:on={value > 0}></i>
				{/each}
			</div>
			{#each memories as [label, values]}
				<div class="row">
					<span>{label}</span>
					<div class="bars">
						{#each values as value}
							<b style:height={`${Math.max(4, (value / peak) * 100)}%`}></b>
						{/each}
					</div>
				</div>
			{/each}
			<p class="note">
				No decay adds every patch into the state and never lets go, so the first pulse is still there at the end. Forgetting half throws the pulse away in two steps. The exponential keep is between them, and it cannot exceed 1 while A is negative.
			</p>
		</article>

		<article>
			<h3>Why the class token is in the middle</h3>
			<p>
				A patch reaches the class token by being carried that many steps. Color is how much of it is left after those steps, using this block's {(keep * 100).toFixed(0)}% keep. A token at the start or the end makes the far corner travel the whole sequence.
			</p>
			<div class="places">
				{#each placements as [label, steps]}
					<figure>
						<div class="board">
							{#each steps as step}
								<i style:background={`rgba(109, 40, 217, ${Math.pow(keep, step)})`} title="{step} steps, {(Math.pow(keep, step) * 100).toFixed(0)}% left"></i>
							{/each}
						</div>
						<figcaption>
							{label}
							<br />
							farthest patch has {(Math.pow(keep, Math.max(...steps)) * 100).toFixed(0)}% left
						</figcaption>
					</figure>
				{/each}
			</div>
			<p class="note">
				ViT sticks the class token on the front. That is fine for attention, which jumps in one step. A scan cannot jump, so Vision Mamba puts the token in the middle and runs both directions. The longest carry is cut from 9 steps to 5.
			</p>
		</article>

		<article>
			<h3>Why a gate, and why a convolution</h3>
			<p>
				SiLU(z) multiplies the scan, channel by channel. It can close a channel. A width-3 convolution lets Δ, B, and C see the neighboring tokens before the step. Take either one away and this picture is classified again.
			</p>
			<div class="paths">
				{#each mechanisms as [label, result]}
					<div>
						<strong>{label}</strong>
						<em style:color={CLASS_COLORS[result.best]}>{classes[result.best]}</em>
						{#each result.probs as probability, index}
							<div class="mini">
								<span>{classes[index]}</span>
								<div><b style:width={`${probability * 100}%`} style:background={CLASS_COLORS[index]}></b></div>
							</div>
						{/each}
					</div>
				{/each}
			</div>
			<p class="note">
				Without the gate, circle and square fall over: every channel of the scan is forced through. Without the convolution, the step size only sees one patch, and the same two shapes fall over. ReLU still classifies these four, because it also blocks negative channels. SiLU is that block without a hard corner.
			</p>
		</article>
	</div>
</section>

<style>
	.mech {
		padding: 8px 16px 18px;
		background: white;
		border-bottom: 1px solid #e2e8f0;
	}
	h2 {
		margin: 4px 0 10px;
		font-size: 15px;
	}
	.grid {
		display: grid;
		grid-template-columns: 1fr 1fr 1.2fr;
		gap: 10px;
	}
	article {
		border: 1px solid #e2e8f0;
		border-radius: 12px;
		padding: 12px;
		background: #f8fafc;
	}
	h3 {
		margin: 0 0 6px;
		font-size: 14px;
	}
	p,
	.note {
		margin: 0 0 8px;
		color: #334155;
		font-size: 13px;
		line-height: 1.45;
	}
	.note {
		color: #64748b;
		margin: 8px 0 0;
	}
	code {
		font-size: 12px;
	}
	.pulse {
		display: flex;
		gap: 4px;
		margin-bottom: 8px;
	}
	.pulse i {
		width: 14px;
		height: 14px;
		border-radius: 3px;
		background: #e2e8f0;
	}
	.pulse i.on {
		background: #7c3aed;
	}
	.row {
		display: grid;
		grid-template-columns: 88px 1fr;
		gap: 8px;
		align-items: end;
		margin-bottom: 6px;
	}
	.row span {
		font-size: 12px;
		color: #475569;
	}
	.bars {
		height: 46px;
		display: flex;
		align-items: flex-end;
		gap: 3px;
	}
	.bars b {
		flex: 1;
		background: #7c3aed;
		border-radius: 3px 3px 0 0;
		display: block;
	}
	.places {
		display: flex;
		gap: 8px;
	}
	figure {
		margin: 0;
		flex: 1;
	}
	.board {
		display: grid;
		grid-template-columns: repeat(3, 22px);
		gap: 3px;
	}
	.board i {
		width: 22px;
		height: 22px;
		border-radius: 4px;
		display: block;
		box-shadow: inset 0 0 0 1px rgb(15 23 42 / 12%);
	}
	figcaption {
		margin-top: 4px;
		font-size: 11px;
		color: #64748b;
		line-height: 1.35;
	}
	.paths {
		display: flex;
		gap: 8px;
	}
	.paths > div {
		flex: 1;
		min-width: 0;
	}
	.paths strong,
	.paths em {
		display: block;
		font-size: 11px;
		font-style: normal;
	}
	.paths em {
		font-weight: 700;
		margin: 2px 0 4px;
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
	.mini div {
		height: 6px;
		background: white;
		border-radius: 99px;
		overflow: hidden;
	}
	.mini b {
		display: block;
		height: 100%;
		border-radius: 99px;
	}
	@media (max-width: 1100px) {
		.grid {
			grid-template-columns: 1fr;
		}
	}
</style>
