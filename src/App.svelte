<script>
	import { onMount } from "svelte";
	import weights from "./lib/weights.json";
	import exampleFile from "./lib/examples.json";
	import { forward, patchFromToken, sampleIndex, softmax, truncate } from "./lib/vim.js";
	import ReadMap from "./components/ReadMap.svelte";
	import Stage from "./components/Stage.svelte";
	import WhyMath from "./components/WhyMath.svelte";
	import Mechanics from "./components/Mechanics.svelte";
	import Factors from "./components/Factors.svelte";
	import { CLASS_COLORS, paintImage } from "./lib/color.js";
	import { PAGES, POPOVERS, hot } from "./lib/pages.js";
	import Flow from "./components/Flow.svelte";
	import Textbook from "./components/Textbook.svelte";
	import Article from "./components/Article.svelte";

	const examples = exampleFile.examples;
	const classes = weights.meta.classes;
	const brushes = [
		[0.93, 0.33, 0.28],
		[0.27, 0.47, 0.95],
		[0.18, 0.72, 0.42],
		[0.95, 0.76, 0.22],
		[0.05, 0.05, 0.05]
	];

	let draft = cloneImage(examples[0].image);
	let exampleId = examples[0].id;
	let brush = 0;
	let stamp = 0;
	let dirty = false;
	let drawing = false;
	let board;
	let run = null;
	let fullLogits = null;
	let probs = [];
	let shifts = [];
	let picked = 0;
	let covered = Array(9).fill(false);
	let modelImage = null;
	let temperature = 1;
	let sampleType = "top-k";
	let sampleValue = 4;
	let blockIdx = 0;
	let direction = 0;
	let focus = null;
	let selected = weights.meta.clsAt;
	let scanT = weights.meta.length - 1;
	let playing = false;
	let page = 0;
	let appliedPage = 0;
	let bookOpen = true;
	let articleOpen = false;
	let popover = null;
	let blendOn = false;
	let blendFrom = 0;
	let blendTo = 1;
	let blendT = 0.5;
	let liveFrame = 0;

	function cloneImage(image) {
		return image.map((channel) => channel.map((row) => row.slice()));
	}

	function sampling() {
		return { type: sampleType, value: Number(sampleValue) };
	}

	function belief(logits) {
		return softmax(logits.map((value) => value / Number(temperature)));
	}

	function updateProbs(resample) {
		if (!run) return;
		const current = belief(run.logits);
		const base = fullLogits ? belief(fullLogits) : current;
		shifts = current.map((value, index) => value - base[index]);
		probs = truncate(current, sampling());
		if (resample || !probs[picked]) picked = sampleIndex(probs);
	}

	function applyCovers(image) {
		const next = cloneImage(image);
		const patch = weights.meta.patch;
		const grid = weights.meta.image / patch;
		covered.forEach((isCovered, index) => {
			if (!isCovered) return;
			const row = Math.floor(index / grid);
			const col = index % grid;
			for (let channel = 0; channel < 3; channel += 1) {
				for (let y = 0; y < patch; y += 1) {
					for (let x = 0; x < patch; x += 1) {
						next[channel][row * patch + y][col * patch + x] = 0.05;
					}
				}
			}
		});
		return next;
	}

	function lerpImage(left, right, amount) {
		const mix = Number(amount);
		return left.map((channel, c) =>
			channel.map((row, y) => row.map((value, x) => value * (1 - mix) + right[c][y][x] * mix))
		);
	}

	function picture() {
		if (!blendOn) return cloneImage(draft);
		return lerpImage(examples[Number(blendFrom)].image, examples[Number(blendTo)].image, blendT);
	}

	function rerun(resetView) {
		const cleanImage = picture();
		const clean = forward(weights, cleanImage);
		fullLogits = clean.logits;
		if (covered.some(Boolean)) {
			modelImage = applyCovers(draft);
			run = forward(weights, modelImage);
		} else {
			modelImage = cleanImage;
			run = clean;
		}
		dirty = false;
		updateProbs(true);
		if (resetView) {
			selected = weights.meta.clsAt;
			scanT = weights.meta.length - 1;
		}
	}

	function classify() {
		rerun(false);
	}

	function finishDraw() {
		drawing = false;
		if (dirty) rerun(false);
	}

	function loadExample(example) {
		blendOn = false;
		draft = cloneImage(example.image);
		exampleId = example.id;
		covered = Array(9).fill(false);
		stamp += 1;
		rerun(true);
	}

	function scheduleLive() {
		if (liveFrame) return;
		liveFrame = requestAnimationFrame(() => {
			liveFrame = 0;
			if (dirty) rerun(false);
		});
	}

	function setBlend(on) {
		blendOn = on;
		if (on) covered = Array(9).fill(false);
		stamp += 1;
		rerun(false);
	}

	function toggleCover(event) {
		const index = event.detail;
		covered = covered.map((value, patch) => (patch === index ? !value : value));
		rerun(false);
	}

	function chooseType(type) {
		sampleType = type;
		sampleValue = type === "top-k" ? 2 : 0.8;
		updateProbs(true);
	}

	function paintAt(event) {
		if (!drawing && event.type !== "pointerdown") return;
		const rect = board.getBoundingClientRect();
		const x = Math.floor(((event.clientX - rect.left) / rect.width) * 24);
		const y = Math.floor(((event.clientY - rect.top) / rect.height) * 24);
		const color = brushes[brush];
		let changed = false;
		for (let dy = 0; dy < 2; dy += 1) {
			for (let dx = 0; dx < 2; dx += 1) {
				const yy = y + dy;
				const xx = x + dx;
				if (yy < 0 || xx < 0 || yy >= 24 || xx >= 24) continue;
				for (let channel = 0; channel < 3; channel += 1) draft[channel][yy][xx] = color[channel];
				changed = true;
			}
		}
		if (!changed) return;
		dirty = true;
		exampleId = "custom";
		stamp += 1;
		scheduleLive();
	}

	function showExample(node, image) {
		paintImage(node, image, 2);
	}

	function showPopover(event) {
		popover = event.detail;
	}

	$: if (board && stamp >= 0) paintImage(board, blendOn ? picture() : draft, 7);
	$: ringPatch = patchFromToken(playing ? scanT : selected, weights.meta.clsAt);
	$: if (page !== appliedPage) {
		appliedPage = page;
		focus = PAGES[page].focus;
	}
	$: sampleMax = sampleType === "top-k" ? 4 : 1;
	$: sampleMin = sampleType === "top-k" ? 1 : 0.1;
	$: sampleStep = sampleType === "top-k" ? 1 : 0.05;

	onMount(() => {
		classify();
		const close = (event) => {
			if (event.target.closest(".popover, .title, .drop, .logit-hit, .play")) return;
			popover = null;
		};
		window.addEventListener("click", close);
		return () => window.removeEventListener("click", close);
	});
</script>

<div class="page">
	<header>
		<div class="brand">
			<strong>Vision Mamba Explainer</strong>
			<span>Tiny bidirectional Vim, running in this page</span>
		</div>
		<div class="header-actions">
			<button type="button" class="ghost" on:click={() => (articleOpen = true)}>Article</button>
			<button type="button" class="ghost" on:click={() => (bookOpen = !bookOpen)}>Textbook</button>
		</div>
	</header>

	<div class="controls">
		<div class="board-wrap" data-col="input" class:cold={focus && !hot(focus, "input")}>
			<div class="canvas-box" class:locked={blendOn}>
				<canvas
					bind:this={board}
					class="board"
					on:pointerdown={(event) => {
						if (blendOn) return;
						drawing = true;
						event.currentTarget.setPointerCapture(event.pointerId);
						paintAt(event);
					}}
					on:pointermove={paintAt}
					on:pointerup={finishDraw}
					on:pointerleave={finishDraw}
				></canvas>
				<div class="ring" aria-hidden="true">
					{#each Array(9) as _, index}
						<i class:on={ringPatch === index}></i>
					{/each}
				</div>
			</div>
			<div class="brushes">
				{#each brushes as color, index}
					<button
						type="button"
						class:on={brush === index}
						style:background={`rgb(${color.map((v) => Math.round(v * 255)).join(",")})`}
						aria-label={index === brushes.length - 1 ? "Erase" : `Brush ${index + 1}`}
						on:click={() => (brush = index)}
					></button>
				{/each}
			</div>
		</div>

		<div class="examples">
			{#each examples as example}
				<button
					type="button"
					class:on={exampleId === example.id}
					on:click={() => loadExample(example)}
				>
					<canvas use:showExample={example.image}></canvas>
					<span>{example.label}</span>
				</button>
			{/each}
		</div>

		<button type="button" class="classify" class:dirty data-col="classify" class:cold={focus && !hot(focus, "classify")} on:click={classify}>
			{dirty ? "Classify drawing" : "Classify"}
		</button>

		{#if run}
			<div class="chip" style:color={CLASS_COLORS[picked]} style:border-color={CLASS_COLORS[picked]}>
				{classes[picked]}
				<small>{(probs[picked] * 100).toFixed(0)}%</small>
			</div>
			<div class="live" aria-hidden="true">
				{#each probs as probability, index}
					<span>
						<b style:width={`${probability * 100}%`} style:background={CLASS_COLORS[index]}></b>
					</span>
				{/each}
			</div>
		{/if}

		<div class="mix">
			<button type="button" class:on={playing} on:click={() => (playing = !playing)}>{playing ? "Pause scan" : "Play scan"}</button>
			<button type="button" class:on={blendOn} on:click={() => setBlend(!blendOn)}>Mix</button>
			{#if blendOn}
				<select bind:value={blendFrom} on:change={() => { stamp += 1; rerun(false); }}>
					{#each examples as example, index}
						<option value={index}>{example.label}</option>
					{/each}
				</select>
				<label>
					<span>{Math.round(Number(blendT) * 100)}%</span>
					<input
						type="range"
						min="0"
						max="1"
						step="0.02"
						bind:value={blendT}
						on:input={() => { stamp += 1; rerun(false); }}
					/>
				</label>
				<select bind:value={blendTo} on:change={() => { stamp += 1; rerun(false); }}>
					{#each examples as example, index}
						<option value={index}>{example.label}</option>
					{/each}
				</select>
			{/if}
		</div>

		<label class="slider" data-col="temperature" class:cold={focus && !hot(focus, "temperature")}>
			<span>Temperature <b>{Number(temperature).toFixed(1)}</b></span>
			<input
				type="range"
				min="0.1"
				max="2"
				step="0.1"
				bind:value={temperature}
				on:input={() => updateProbs(false)}
				on:change={() => updateProbs(true)}
			/>
		</label>

		<div class="sampling" data-col="sampling" class:cold={focus && !hot(focus, "sampling")}>
			<div class="modes">
				<span>Sampling</span>
				<button type="button" class:on={sampleType === "top-k"} on:click={() => chooseType("top-k")}>Top-k</button>
				<button type="button" class:on={sampleType === "top-p"} on:click={() => chooseType("top-p")}>Top-p</button>
				<button type="button" on:click={() => updateProbs(true)}>Draw class</button>
			</div>
			<label>
				<span>{sampleType === "top-k" ? "k" : "p"} = {Number(sampleValue).toFixed(sampleType === "top-k" ? 0 : 2)}</span>
				<input
					type="range"
					min={sampleMin}
					max={sampleMax}
					step={sampleStep}
					bind:value={sampleValue}
					on:input={() => updateProbs(false)}
					on:change={() => updateProbs(true)}
				/>
			</label>
		</div>
	</div>

	{#if modelImage}
		<Stage image={modelImage} {classes} bind:blockIdx />
	{/if}

	<div class="workspace">
		<div class="main-col">
			{#if run}
				<ReadMap
					image={draft}
					{stamp}
					{run}
					{blockIdx}
					{selected}
					{covered}
					on:select={(event) => (selected = event.detail)}
					on:cover={toggleCover}
					on:clear={() => {
						covered = Array(9).fill(false);
						rerun(false);
					}}
					on:block={(event) => (blockIdx = event.detail)}
				/>
				{#if modelImage}
					<WhyMath image={modelImage} {classes} />
					<Mechanics image={modelImage} {classes} />
					<Factors image={modelImage} {classes} />
				{/if}
			{/if}
			<Flow
				{run}
				{probs}
				{shifts}
				{picked}
				{classes}
				colors={CLASS_COLORS}
				bind:blockIdx
				bind:direction
				bind:focus
				bind:selected
				bind:scanT
				bind:playing
				on:popover={showPopover}
			/>
		</div>
		<Textbook bind:page bind:open={bookOpen} />
	</div>

	<Article bind:open={articleOpen} />

	{#if popover && POPOVERS[popover.id]}
		<div class="popover" style:left="{popover.x}px" style:top="{popover.y}px" role="dialog">
			<h3>{POPOVERS[popover.id].title}</h3>
			<div>{@html POPOVERS[popover.id].html}</div>
		</div>
	{/if}
</div>

<style>
	.page {
		min-height: 100vh;
		display: flex;
		flex-direction: column;
		background: #f8fafc;
		color: #0f172a;
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 16px;
		background: #172554;
		color: white;
		padding: 10px 18px;
	}
	.brand {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.brand span,
	.header-actions .ghost {
		color: #bfdbfe;
		font-size: 12px;
	}
	.header-actions {
		display: flex;
		gap: 8px;
	}
	.ghost,
	.classify,
	.examples button,
	.modes button,
	.brushes button {
		font: inherit;
		cursor: pointer;
	}
	.ghost {
		background: transparent;
		border: 1px solid #1e3a8a;
		border-radius: 8px;
		padding: 6px 10px;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 16px;
		padding: 12px 16px;
		background: white;
		border-bottom: 1px solid #e2e8f0;
	}
	.controls .cold {
		opacity: 0.35;
	}
	.board-wrap {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.canvas-box {
		position: relative;
		width: 168px;
		height: 168px;
	}
	.canvas-box.locked .board {
		cursor: default;
	}
	.board {
		width: 168px;
		height: 168px;
		image-rendering: pixelated;
		border-radius: 8px;
		cursor: crosshair;
		touch-action: none;
		box-shadow: inset 0 0 0 1px #e2e8f0;
	}
	.ring {
		position: absolute;
		inset: 0;
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		pointer-events: none;
	}
	.ring i {
		display: block;
	}
	.ring i.on {
		box-shadow: inset 0 0 0 3px #d4a017;
		background: rgb(212 160 23 / 28%);
	}
	.live {
		display: flex;
		flex-direction: column;
		gap: 3px;
		width: 72px;
	}
	.live span {
		display: block;
		height: 6px;
		background: #e2e8f0;
		border-radius: 99px;
		overflow: hidden;
	}
	.live b {
		display: block;
		height: 100%;
		border-radius: 99px;
		transition: width 180ms ease;
	}
	.mix {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 12px;
		color: #475569;
	}
	.mix button,
	.mix select {
		font: inherit;
		border: 1px solid #e2e8f0;
		background: white;
		border-radius: 999px;
		padding: 4px 10px;
		cursor: pointer;
	}
	.mix button.on {
		background: #172554;
		border-color: #172554;
		color: white;
	}
	.mix label {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.mix input {
		width: 120px;
	}
	.brushes {
		display: flex;
		gap: 4px;
	}
	.brushes button {
		width: 18px;
		height: 18px;
		border-radius: 99px;
		border: 2px solid white;
		box-shadow: 0 0 0 1px #cbd5e1;
	}
	.brushes button.on {
		box-shadow: 0 0 0 2px #172554;
	}
	.examples {
		display: flex;
		gap: 8px;
	}
	.examples button {
		border: 1px solid #e2e8f0;
		background: white;
		border-radius: 10px;
		padding: 6px;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
		color: #475569;
		font-size: 12px;
	}
	.examples button.on {
		border-color: #172554;
		color: #172554;
	}
	.examples canvas {
		width: 48px;
		height: 48px;
		image-rendering: pixelated;
	}
	.classify {
		border: 0;
		background: #172554;
		color: white;
		border-radius: 10px;
		padding: 10px 16px;
		font-weight: 650;
	}
	.classify.dirty {
		background: #7c3aed;
	}
	.chip {
		border: 1px solid;
		border-radius: 999px;
		padding: 6px 12px;
		font-weight: 700;
		display: flex;
		align-items: baseline;
		gap: 6px;
	}
	.chip small {
		font-weight: 500;
		color: #64748b;
	}
	.slider,
	.sampling {
		display: flex;
		flex-direction: column;
		gap: 4px;
		min-width: 180px;
		font-size: 12px;
		color: #475569;
	}
	.modes {
		display: flex;
		gap: 6px;
		align-items: center;
	}
	.modes button {
		border: 1px solid #e2e8f0;
		background: white;
		border-radius: 999px;
		padding: 3px 8px;
	}
	.modes button.on {
		background: #ede9fe;
		border-color: #7c3aed;
		color: #5b21b6;
	}
	.workspace {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 320px;
		align-items: stretch;
		flex: 1;
	}
	.main-col {
		min-width: 0;
	}
	.popover {
		position: fixed;
		z-index: 50;
		width: 300px;
		background: white;
		border: 1px solid #e2e8f0;
		border-radius: 12px;
		padding: 12px 14px;
		box-shadow: 0 16px 40px rgb(15 23 42 / 12%);
	}
	.popover h3 {
		margin: 0 0 6px;
		font-size: 15px;
	}
	.popover :global(p) {
		margin: 0 0 8px;
		color: #334155;
		font-size: 13px;
		line-height: 1.45;
	}
	.popover :global(pre) {
		background: #f8fafc;
		border-radius: 8px;
		padding: 8px;
		font-size: 12px;
		overflow-x: auto;
	}
	.popover :global(.mono) {
		font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
	}
	@media (max-width: 1100px) {
		.workspace {
			grid-template-columns: 1fr;
		}
	}
</style>
