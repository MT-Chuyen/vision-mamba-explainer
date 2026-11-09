<script>
	import { onDestroy } from "svelte";
	import weights from "../lib/weights.json";
	import { classify } from "../lib/vim.js";
	import { CLASS_COLORS } from "../lib/color.js";

	export let image;
	export let classes = [];
	export let blockIdx = 0;

	let depthPlay = false;
	let timer = null;

	$: frames = image
		? weights.blocks.map((_, index) => classify(weights, image, { layers: index + 1 }))
		: [];
	$: current = frames[blockIdx];

	function stop() {
		if (timer) clearInterval(timer);
		timer = null;
		depthPlay = false;
	}

	function toggle() {
		if (depthPlay) {
			stop();
			return;
		}
		depthPlay = true;
		timer = setInterval(() => {
			blockIdx = (blockIdx + 1) % weights.blocks.length;
		}, 900);
	}

	onDestroy(stop);
</script>

<section class="stage" data-col="stage">
	<div class="head">
		<div>
			<h2>Watch the answer form</h2>
			<p>
				{#if current}
					After block {blockIdx + 1}, <b style:color={CLASS_COLORS[current.best]}>{classes[current.best]}</b>
					leads at {(current.probs[current.best] * 100).toFixed(0)}%. Each frame keeps only the first blocks and runs the same picture again.
				{/if}
			</p>
		</div>
		<button type="button" class:on={depthPlay} on:click={toggle}>{depthPlay ? "Pause" : "Play depth"}</button>
	</div>
	<div class="frames">
		{#each frames as frame, index}
			<button type="button" class:on={index === blockIdx} on:click={() => (blockIdx = index)}>
				<strong>Block {index + 1}</strong>
				<em style:color={CLASS_COLORS[frame.best]}>{classes[frame.best]}</em>
				{#each frame.probs as probability, classIndex}
					<span>
						<i style:width={`${probability * 100}%`} style:background={CLASS_COLORS[classIndex]}></i>
					</span>
				{/each}
			</button>
		{/each}
	</div>
</section>

<style>
	.stage {
		padding: 12px 16px 14px;
		background: white;
		border-bottom: 1px solid #e2e8f0;
	}
	.head {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		align-items: flex-start;
	}
	h2 {
		margin: 0 0 4px;
		font-size: 15px;
	}
	p {
		margin: 0;
		color: #334155;
		font-size: 13px;
		line-height: 1.45;
		max-width: 720px;
	}
	button {
		font: inherit;
		cursor: pointer;
	}
	.head button {
		border: 1px solid #172554;
		background: #172554;
		color: white;
		border-radius: 999px;
		padding: 6px 12px;
		white-space: nowrap;
	}
	.head button.on {
		background: white;
		color: #172554;
	}
	.frames {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 8px;
		margin-top: 10px;
	}
	.frames button {
		text-align: left;
		border: 1px solid #e2e8f0;
		background: #f8fafc;
		border-radius: 12px;
		padding: 8px;
	}
	.frames button.on {
		border-color: #172554;
		box-shadow: inset 0 0 0 1px #172554;
		background: white;
	}
	strong,
	em {
		display: block;
		font-style: normal;
	}
	strong {
		font-size: 11px;
		color: #64748b;
		font-weight: 600;
	}
	em {
		font-size: 13px;
		font-weight: 700;
		margin: 2px 0 6px;
	}
	.frames span {
		display: block;
		height: 6px;
		background: #e2e8f0;
		border-radius: 99px;
		overflow: hidden;
		margin-top: 3px;
	}
	.frames i {
		display: block;
		height: 100%;
		border-radius: 99px;
		transition: width 280ms ease;
	}
	@media (max-width: 800px) {
		.frames {
			grid-template-columns: 1fr 1fr;
		}
	}
</style>
