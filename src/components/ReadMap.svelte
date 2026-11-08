<script>
	import { createEventDispatcher } from "svelte";
	import { mixedPatches, patchFromToken, patchesFromRow } from "../lib/vim.js";

	export let image;
	export let stamp = 0;
	export let run;
	export let blockIdx = 0;
	export let selected = 4;
	export let covered = [];

	const dispatch = createEventDispatcher();
	let mode = "both";
	let tool = "look";
	let canvas;

	$: clsAt = run.tokens.findIndex((token) => token.cls);

	function valuesFor(block, which, token, at) {
		if (which === "forward") return patchesFromRow(block.forward.influence[token], at);
		if (which === "backward") return patchesFromRow(block.backward.influence[token], at);
		return mixedPatches(block.forward.influence[token], block.backward.influence[token], at);
	}

	function classValues(block) {
		const index = run.tokens.findIndex((token) => token.cls);
		return mixedPatches(block.forward.influence[index], block.backward.influence[index], clsAt);
	}

	function draw(node, patchValues, scale) {
		const patch = 8;
		const grid = 3;
		node.width = patch * grid * scale;
		node.height = patch * grid * scale;
		const ctx = node.getContext("2d");
		for (let y = 0; y < 24; y += 1) {
			for (let x = 0; x < 24; x += 1) {
				const red = Math.round(image[0][y][x] * 255);
				const green = Math.round(image[1][y][x] * 255);
				const blue = Math.round(image[2][y][x] * 255);
				ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
				ctx.fillRect(x * scale, y * scale, scale, scale);
			}
		}
		const cell = patch * scale;
		patchValues.forEach((value, index) => {
			const row = Math.floor(index / grid);
			const col = index % grid;
			const x = col * cell;
			const y = row * cell;
			ctx.fillStyle = `rgba(109, 40, 217, ${Math.min(0.78, value * 0.85)})`;
			ctx.fillRect(x, y, cell, cell);
			ctx.strokeStyle = "rgba(255,255,255,0.75)";
			ctx.lineWidth = 1;
			ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
			if (covered[index]) {
				ctx.strokeStyle = "#0f172a";
				ctx.lineWidth = 2;
				ctx.beginPath();
				ctx.moveTo(x + 6, y + 6);
				ctx.lineTo(x + cell - 6, y + cell - 6);
				ctx.moveTo(x + cell - 6, y + 6);
				ctx.lineTo(x + 6, y + cell - 6);
				ctx.stroke();
			}
			if (value >= 0.08) {
				ctx.fillStyle = "#fff";
				ctx.font = `${Math.max(10, scale * 2)}px sans-serif`;
				ctx.textAlign = "center";
				ctx.fillText(String(Math.round(value * 100)), x + cell / 2, y + cell / 2 + 4);
			}
		});
		if (selectedPatch >= 0) {
			const row = Math.floor(selectedPatch / grid);
			const col = selectedPatch % grid;
			ctx.strokeStyle = "#d4a017";
			ctx.lineWidth = 3;
			ctx.strokeRect(col * cell + 2, row * cell + 2, cell - 4, cell - 4);
		}
	}

	function paintFrame(node, patchValues) {
		const render = (next) => draw(node, next, 2);
		render(patchValues);
		return { update: render };
	}

	function hit(event) {
		const rect = event.currentTarget.getBoundingClientRect();
		const col = Math.min(2, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * 3)));
		const row = Math.min(2, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * 3)));
		const patch = row * 3 + col;
		if (tool === "hide") dispatch("cover", patch);
		else dispatch("select", patch < clsAt ? patch : patch + 1);
	}

	$: block = run?.blocks[blockIdx];
	$: selectedPatch = patchFromToken(selected, clsAt);
	$: shown = block ? valuesFor(block, mode, selected, clsAt) : [];
	$: leader = shown.reduce(
		(best, value, index) => (value > best.value ? { value, index } : best),
		{ value: -1, index: 0 }
	);
	$: otherMass = shown.reduce((total, value, index) => (index === selectedPatch ? total : total + value), 0);
	$: step = block
		? (block.forward.delta[selected].reduce((total, value) => total + value, 0) +
				block.backward.delta[selected].reduce((total, value) => total + value, 0)) /
			(2 * block.forward.delta[selected].length)
		: 0;
	$: if (canvas && image && shown.length && stamp >= 0 && covered) draw(canvas, shown, 6);
</script>

<section class="read" data-col="readmap">
	<div class="map-wrap">
		<canvas bind:this={canvas} on:click={hit}></canvas>
		<div class="tools">
			<button type="button" class:on={tool === "look"} on:click={() => (tool = "look")}>Look</button>
			<button type="button" class:on={tool === "hide"} on:click={() => (tool = "hide")}>Hide</button>
			<button type="button" on:click={() => dispatch("clear")}>Clear hides</button>
		</div>
	</div>
	<div class="copy">
		<div class="modes">
			{#each ["both", "forward", "backward"] as item}
				<button type="button" class:on={mode === item} on:click={() => (mode = item)}>
					{item === "both" ? "Both" : item === "forward" ? "Forward" : "Backward"}
				</button>
			{/each}
		</div>
		<h2>
			{run.tokens[selected].cls ? "Class token" : `Patch ${run.tokens[selected].label}`}
			<span>block {blockIdx + 1}</span>
		</h2>
		<p>
			{#if selectedPatch < 0}
				Strongest patch is {Math.floor(leader.index / 3)},{leader.index % 3} at {(leader.value * 100).toFixed(0)}%.
			{:else}
				{(otherMass * 100).toFixed(0)}% comes from other patches. The strongest is
				{Math.floor(leader.index / 3)},{leader.index % 3} at {(leader.value * 100).toFixed(0)}%.
			{/if}
			Mean Δ is {step.toFixed(2)}. A larger Δ forgets the state sooner.
		</p>
		<div class="film">
			{#each run.blocks as item, index}
				<button type="button" class:on={index === blockIdx} on:click={() => dispatch("block", index)}>
					<canvas use:paintFrame={classValues(item)}></canvas>
					<span>{index + 1}</span>
				</button>
			{/each}
		</div>
		<p class="note">The four frames are the class token, both directions. Click one to open that block. Hide blanks a patch and reruns.</p>
	</div>
</section>

<style>
	.read {
		display: flex;
		gap: 16px;
		align-items: flex-start;
		padding: 12px 16px 4px;
		background: white;
		border-bottom: 1px solid #e2e8f0;
	}
	canvas {
		width: 144px;
		height: 144px;
		image-rendering: pixelated;
		border-radius: 8px;
		cursor: pointer;
		display: block;
	}
	.tools,
	.modes,
	.film {
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
	}
	.tools {
		margin-top: 6px;
	}
	button {
		font: inherit;
		border: 1px solid #e2e8f0;
		background: white;
		border-radius: 999px;
		padding: 3px 8px;
		cursor: pointer;
		color: #475569;
		font-size: 12px;
	}
	button.on {
		background: #ede9fe;
		border-color: #7c3aed;
		color: #5b21b6;
	}
	.copy {
		display: flex;
		flex-direction: column;
		gap: 8px;
		min-width: 0;
	}
	h2 {
		margin: 0;
		font-size: 16px;
		display: flex;
		gap: 8px;
		align-items: baseline;
	}
	h2 span,
	.note {
		color: #94a3b8;
		font-size: 12px;
		font-weight: 500;
	}
	p {
		margin: 0;
		color: #334155;
		font-size: 13px;
		line-height: 1.45;
		max-width: 460px;
	}
	.film button {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 2px;
		border-radius: 8px;
		padding: 4px;
	}
	.film canvas {
		width: 48px;
		height: 48px;
		cursor: pointer;
	}
	@media (max-width: 800px) {
		.read {
			flex-direction: column;
		}
	}
</style>
