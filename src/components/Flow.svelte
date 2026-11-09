<script>
	import { createEventDispatcher, onDestroy, onMount, tick } from "svelte";
	import VectorBar from "./VectorBar.svelte";
	import { heat, paintThumb } from "../lib/color.js";
	import { hot } from "../lib/pages.js";

	export let run;
	export let blockIdx = 0;
	export let direction = 0;
	export let focus = null;
	export let selected = 4;
	export let scanT = 9;
	export let probs = [];
	export let shifts = [];
	export let picked = 0;
	export let classes = [];
	export let colors = [];
	export let playing = false;

	let root;
	let paths = [];
	let frame = 0;
	let timer;
	let tip = null;

	$: block = run?.blocks[blockIdx];
	$: scan = block ? (direction === 0 ? block.forward : block.backward) : null;
	$: length = run?.tokens.length ?? 0;
	$: remaining = run ? run.blocks.length - blockIdx - 1 : 0;

	function ribbon(x1, y1, x2, y2, width) {
		const bend = Math.min(90, Math.abs(x2 - x1) * 0.45);
		const half = Math.max(2, width / 2);
		return `M ${x1} ${y1 - half}
			C ${x1 + bend} ${y1 - half}, ${x2 - bend} ${y2 - half}, ${x2} ${y2 - half}
			L ${x2} ${y2 + half}
			C ${x2 - bend} ${y2 + half}, ${x1 + bend} ${y1 + half}, ${x1} ${y1 + half} Z`;
	}

	function anchors(name, box) {
		return [...root.querySelectorAll(`[data-anchor="${name}"]`)].map((node) => {
			const rect = node.getBoundingClientRect();
			return {
				index: Number(node.dataset.index),
				right: rect.right - box.left,
				left: rect.left - box.left,
				y: rect.top + rect.height / 2 - box.top,
				width: rect.width
			};
		});
	}

	async function measure() {
		await tick();
		if (!root || !run) return;
		const box = root.getBoundingClientRect();
		const links = [
			["sum", "stream", "#93c5fd"],
			["out", "more", "#c4b5fd"],
			["cls-more", "logit", "#ddd6fe"]
		];
		const next = [];
		for (const [from, to, color] of links) {
			const sources = anchors(from, box);
			const targets = anchors(to, box);
			for (const source of sources) {
				const target = targets.find((item) => item.index === source.index);
				if (!target || source.width === 0 || target.width === 0) continue;
				next.push({
					color,
					hot: source.index === selected || from === "cls-more",
					d: ribbon(
						source.right,
						source.y,
						target.left,
						target.y,
						source.index === selected ? 10 : 6
					)
				});
			}
		}
		paths = next;
	}

	function schedule() {
		cancelAnimationFrame(frame);
		frame = requestAnimationFrame(() => measure());
	}

	function choose(index) {
		selected = index;
		schedule();
	}

	function stepBlock(delta) {
		if (!run) return;
		blockIdx = (blockIdx + delta + run.blocks.length) % run.blocks.length;
	}

	function syncPlay(active) {
		if (active && !timer) {
			timer = setInterval(() => {
				scanT = length ? (scanT + 1) % length : 0;
			}, 700);
		}
		if (!active && timer) {
			clearInterval(timer);
			timer = null;
		}
	}

	function togglePlay() {
		playing = !playing;
	}

	$: syncPlay(playing);

	const FOCUS = {
		patches: "patches",
		embedding: "embedding",
		position: "position",
		residual: "residual",
		norm: "norm",
		x: "xz",
		delta: "dbc",
		b: "dbc",
		c: "dbc",
		scan: "scan",
		gate: "gate",
		drop: "drop",
		logits: "logits",
		softmax: "probs"
	};

	const dispatch = createEventDispatcher();

	function open(id, event) {
		event.stopPropagation();
		const next = FOCUS[id] || id;
		focus = focus === next ? null : next;
		dispatchPopover(id, event);
	}

	function dispatchPopover(id, event) {
		const rect = event.currentTarget.getBoundingClientRect();
		dispatch("popover", { id, x: Math.min(rect.left, window.innerWidth - 320), y: rect.bottom + 8 });
	}

	function masked(row, col) {
		return direction === 0 ? col > row : col < row;
	}

	function followTip(event) {
		const node = event.target.closest?.("[data-tip]");
		if (!node) {
			tip = null;
			return;
		}
		tip = { text: node.dataset.tip, x: event.clientX + 14, y: event.clientY + 16 };
	}

	$: if (root && run && blockIdx >= 0 && direction >= 0 && selected >= 0) schedule();

	onMount(() => {
		const observer = new ResizeObserver(() => schedule());
		observer.observe(root);
		return () => observer.disconnect();
	});

	onDestroy(() => {
		if (timer) clearInterval(timer);
	});
</script>

<div class="scroll">
	<div
		class="flow"
		bind:this={root}
		role="group"
		on:mousemove={followTip}
		on:mouseleave={() => (tip = null)}
	>
		<svg class="links" aria-hidden="true">
			{#each paths as path}
				<path d={path.d} fill={path.color} opacity={path.hot ? 0.8 : 0.28} />
			{/each}
		</svg>

		{#if run && block && scan}
			<section class="col" data-col="input" class:cold={focus && !hot(focus, "input")}>
				<button class="title" type="button" on:click={(event) => open("patches", event)}>Patches</button>
				<div class="stack">
					{#each run.tokens as token, index}
						<button
							type="button"
							class="cell name"
							class:on={selected === index}
							class:cls={token.cls}
							on:click={() => choose(index)}
						>
							<span>{token.label}</span>
							{#if token.thumb}
								<canvas use:paintThumb={token.thumb} class="thumb" data-tip="patch {token.label}"></canvas>
							{:else}
								<span class="cls-pill" data-tip="class token, middle of the sequence">CLS</span>
							{/if}
						</button>
					{/each}
				</div>
			</section>

			<section class="col" data-col="embed" class:cold={focus && !hot(focus, "embed")}>
				<button class="title" type="button" on:click={(event) => open("embedding", event)}>Embedding</button>
				<div class="stack">
					{#each run.tokens as token, index}
						<div class="cell" class:on={selected === index} class:cls={token.cls}>
							<VectorBar
								values={token.embedding}
								hue={token.cls ? 43 : 217}
								anchor="embed"
								{index}
								tip={`embedding (${token.embedding.length})`}
							/>
						</div>
					{/each}
				</div>
			</section>

			<section class="col" data-col="pos" class:cold={focus && !hot(focus, "pos")}>
				<button class="title" type="button" on:click={(event) => open("position", event)}>Position</button>
				<div class="stack">
					{#each run.tokens as token, index}
						<div class="cell" class:on={selected === index} class:cls={token.cls}>
							<span class="op">+</span>
							<VectorBar values={token.position} hue={199} {index} tip={`position (${token.position.length})`} />
							<span class="op">=</span>
							<VectorBar
								values={token.sum}
								hue={217}
								anchor="sum"
								{index}
								tip={`embedding + position (${token.sum.length})`}
							/>
						</div>
					{/each}
				</div>
			</section>

			<section class="col block" data-col="stream" class:cold={focus && !hot(focus, "stream")}>
				<div class="title-row">
					<button class="title" type="button" on:click={(event) => open("residual", event)}>
						Block {blockIdx + 1}/{run.blocks.length}
					</button>
					<span class="pager">
						<button type="button" on:click={() => stepBlock(-1)} aria-label="Previous block">‹</button>
						<button type="button" on:click={() => stepBlock(1)} aria-label="Next block">›</button>
					</span>
				</div>
				<div class="stack">
					{#each block.input as vector, index}
						<div class="cell" class:on={selected === index} class:cls={run.tokens[index].cls}>
							<VectorBar values={vector} hue={217} anchor="stream" {index} tip={`residual stream (${vector.length})`} />
						</div>
					{/each}
				</div>
			</section>

			<section class="col" data-col="norm" class:cold={focus && !hot(focus, "norm")}>
				<button class="title" type="button" on:click={(event) => open("norm", event)}>Norm</button>
				<div class="stack">
					{#each block.normed as vector, index}
						<div class="cell" class:on={selected === index} class:cls={run.tokens[index].cls}>
							<VectorBar values={vector} hue={215} {index} tip={`layer norm (${vector.length})`} />
						</div>
					{/each}
				</div>
			</section>

			<section class="col" data-col="x" class:cold={focus && !hot(focus, "x")}>
				<button class="title" type="button" on:click={(event) => open("x", event)}>x</button>
				<div class="stack">
					{#each scan.conv as vector, index}
						<div class="cell" class:on={selected === index} class:cls={run.tokens[index].cls}>
							<VectorBar values={vector} hue={217} {index} tip={`convolved x (${vector.length})`} />
						</div>
					{/each}
				</div>
			</section>

			<section class="col" data-col="delta" class:cold={focus && !hot(focus, "delta")}>
				<button class="title" type="button" on:click={(event) => open("delta", event)}>Δ</button>
				<div class="stack">
					{#each scan.delta as vector, index}
						<div class="cell" class:on={selected === index} class:cls={run.tokens[index].cls}>
							<VectorBar values={vector} hue={24} {index} tip={`delta (${vector.length}), positive step size`} />
						</div>
					{/each}
				</div>
			</section>

			<section class="col" data-col="b" class:cold={focus && !hot(focus, "b")}>
				<button class="title" type="button" on:click={(event) => open("b", event)}>B</button>
				<div class="stack">
					{#each scan.b as vector, index}
						<div class="cell" class:on={selected === index} class:cls={run.tokens[index].cls}>
							<VectorBar values={vector} hue={0} {index} tip={`B (${vector.length}), write into the state`} />
						</div>
					{/each}
				</div>
			</section>

			<section class="col" data-col="c" class:cold={focus && !hot(focus, "c")}>
				<button class="title" type="button" on:click={(event) => open("c", event)}>C</button>
				<div class="stack">
					{#each scan.c as vector, index}
						<div class="cell" class:on={selected === index} class:cls={run.tokens[index].cls}>
							<VectorBar values={vector} hue={152} {index} tip={`C (${vector.length}), read from the state`} />
						</div>
					{/each}
				</div>
			</section>

			<section class="col scan" data-col="scan" class:cold={focus && !hot(focus, "scan")}>
				<div class="title-row">
					<button class="title" type="button" on:click={(event) => open("scan", event)}>
						{direction === 0 ? "Forward scan" : "Backward scan"}
					</button>
					<button type="button" class="play" on:click={togglePlay}>{playing ? "Pause" : "Play"}</button>
				</div>
				<div class="dirs">
					{#each [block.forward, block.backward] as item, dir}
						<button
							type="button"
							class:on={direction === dir}
							on:click={() => (direction = dir)}
						>
							<span>{dir === 0 ? "Forward" : "Backward"}</span>
							<span class="mini">
								{#each item.influence[run.tokens.findIndex((token) => token.cls)] as value}
									<i style:background={heat(value)}></i>
								{/each}
							</span>
						</button>
					{/each}
				</div>
				<div class="matrix" style={`--n:${length}`}>
					{#each scan.influence as row, t}
						{#each row as value, i}
							<button
								type="button"
								class:off={masked(t, i)}
								class:now={t === scanT}
								class:pick={i === selected}
								style:background={masked(t, i) ? undefined : heat(value)}
								data-tip={`${run.tokens[t].label} ← ${run.tokens[i].label}: ${(value * 100).toFixed(0)}%`}
								on:click={() => {
									choose(i);
									scanT = t;
								}}
							>
								{#if t === scanT && !masked(t, i)}
									<span>{Math.round(value * 100)}</span>
								{/if}
							</button>
						{/each}
					{/each}
				</div>
				<div class="state">
					<span>state at {run.tokens[scanT].label}</span>
					<VectorBar values={scan.state[scanT]} hue={262} tip="RMS of the hidden state, 4 channels" />
				</div>
				<p class="formula">hₜ = exp(Δₜ A) hₜ₋₁ + Δₜ Bₜ xₜ</p>
			</section>

			<section class="col" data-col="gate" class:cold={focus && !hot(focus, "gate")}>
				<button class="title" type="button" on:click={(event) => open("gate", event)}>Gate</button>
				<div class="stack">
					{#each block.gate as vector, index}
						<div class="cell" class:on={selected === index} class:cls={run.tokens[index].cls}>
							<VectorBar values={vector} hue={280} {index} tip={`SiLU(z) (${vector.length})`} />
						</div>
					{/each}
				</div>
			</section>

			<section class="col" data-col="out" class:cold={focus && !hot(focus, "out")}>
				<button class="title" type="button" on:click={(event) => open("residual", event)}>Residual</button>
				<button class="drop" type="button" on:click={(event) => open("drop", event)}>drop path off</button>
				<div class="stack">
					{#each block.output as vector, index}
						<div class="cell" class:on={selected === index} class:cls={run.tokens[index].cls}>
							<VectorBar values={vector} hue={217} anchor="out" {index} tip={`stream + block (${vector.length})`} />
						</div>
					{/each}
				</div>
			</section>

			<section class="col" data-col="more" class:cold={focus && !hot(focus, "more")}>
				<button class="title more" type="button" on:click={() => stepBlock(1)}>
					{#if remaining > 0}
						{remaining} more identical<br />Vim block{remaining === 1 ? "" : "s"}
					{:else}
						last block
					{/if}
				</button>
				<div class="stack">
					{#each block.output as vector, index}
						<div class="cell" class:on={selected === index} class:cls={run.tokens[index].cls}>
							<div data-anchor="more" data-index={index}>
								{#if run.tokens[index].cls}
									<div data-anchor="cls-more" data-index={index}>
										<VectorBar
											values={vector}
											hue={217}
											{index}
											tip={`block ${blockIdx + 1} output (${vector.length})`}
										/>
									</div>
								{:else}
									<VectorBar
										values={vector}
										hue={217}
										{index}
										tip={`block ${blockIdx + 1} output (${vector.length})`}
									/>
								{/if}
							</div>
						</div>
					{/each}
				</div>
			</section>

			<section class="col probs" data-col="probs" class:cold={focus && !hot(focus, "probs")}>
				<button class="title" type="button" on:click={(event) => open("softmax", event)}>Probabilities</button>
				<div class="readout">
					<button type="button" class="logit-hit" on:click={(event) => open("logits", event)}>
						<VectorBar
							values={run.logits}
							hue={262}
							anchor="logit"
							index={run.tokens.findIndex((token) => token.cls)}
							tip={`class token logits (${run.logits.length})`}
						/>
					</button>
					<span class="from-cls">from CLS</span>
				</div>
				<div class="bars">
					{#each probs as probability, index}
						<div class="bar-row" class:picked={index === picked}>
							<span class="swatch" style:background={colors[index]}></span>
							<span class="cname">{classes[index]}</span>
							<div class="track">
								<div class="fill" style:width={`${probability * 100}%`} style:background={colors[index]}></div>
							</div>
							<b>{(probability * 100).toFixed(1)}%</b>
							{#if Math.abs(shifts[index] || 0) >= 0.005}
								<em class:up={shifts[index] > 0}>
									{shifts[index] > 0 ? "+" : ""}{(shifts[index] * 100).toFixed(0)}
								</em>
							{/if}
						</div>
					{/each}
					<p class="hint">
						Outlined row is the sampled class. Bars sum to 1 after top-k / top-p.
						{#if shifts.some((value) => Math.abs(value) >= 0.005)}
							The small number is the change from covering a patch.
						{/if}
					</p>
				</div>
			</section>
		{/if}
	</div>
	{#if tip}
		<div class="tip" style:left="{tip.x}px" style:top="{tip.y}px">{tip.text}</div>
	{/if}
</div>

<style>
	.scroll {
		overflow-x: auto;
		padding: 8px 8px 28px;
	}
	.flow {
		position: relative;
		display: flex;
		align-items: flex-start;
		gap: 14px;
		min-width: max-content;
		padding: 8px 12px 12px;
	}
	.links {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
		overflow: visible;
	}
	.col {
		position: relative;
		z-index: 1;
		transition: opacity 0.25s ease;
	}
	.col.cold {
		opacity: 0.28;
	}
	.title,
	.drop,
	.play,
	.pager button,
	.dirs button {
		font: inherit;
		color: #64748b;
	}
	.title {
		display: block;
		border: 0;
		background: transparent;
		padding: 0 0 8px;
		cursor: pointer;
		text-align: left;
		font-size: 13px;
		min-height: 34px;
	}
	.title:hover {
		color: #0f172a;
	}
	.title-row {
		display: flex;
		align-items: flex-end;
		justify-content: space-between;
		gap: 8px;
		min-height: 34px;
	}
	.title-row .title {
		min-height: 0;
	}
	.title.more {
		max-width: 110px;
		line-height: 1.25;
	}
	.pager {
		display: flex;
		gap: 4px;
		padding-bottom: 8px;
	}
	.pager button,
	.play,
	.drop {
		border: 1px solid #e2e8f0;
		background: #fff;
		border-radius: 6px;
		cursor: pointer;
	}
	.pager button {
		width: 22px;
		height: 22px;
	}
	.play,
	.drop {
		font-size: 11px;
		padding: 2px 6px;
		margin-bottom: 8px;
	}
	.stack {
		display: flex;
		flex-direction: column;
		gap: var(--gap);
	}
	.cell {
		height: var(--row);
		display: flex;
		align-items: center;
		gap: 4px;
		border-radius: 6px;
	}
	.cell.on {
		background: #eef2ff;
	}
	.cell.cls {
		box-shadow: inset 2px 0 0 #d4a017;
	}
	.cell.name {
		border: 0;
		background: transparent;
		cursor: pointer;
		font: inherit;
		width: 100%;
		justify-content: flex-end;
		padding: 0 4px;
	}
	.cell.name span {
		font-size: 12px;
		color: #334155;
		min-width: 28px;
		text-align: right;
	}
	.thumb {
		width: var(--row);
		height: var(--row);
		image-rendering: pixelated;
		border-radius: 4px;
	}
	.cls-pill {
		width: var(--row);
		height: var(--row);
		border-radius: 4px;
		display: grid;
		place-items: center;
		background: #fef3c7;
		color: #92400e;
		font-size: 10px;
		font-weight: 700;
	}
	.op {
		color: #94a3b8;
		font-size: 12px;
		width: 10px;
		text-align: center;
	}
	.dirs {
		display: flex;
		gap: 6px;
		margin-bottom: 6px;
	}
	.dirs button {
		border: 1px solid #e2e8f0;
		background: #fff;
		border-radius: 8px;
		padding: 4px 6px;
		cursor: pointer;
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 11px;
	}
	.dirs button.on {
		border-color: #7c3aed;
		color: #5b21b6;
	}
	.mini {
		display: flex;
		height: 8px;
		width: 92px;
		border-radius: 3px;
		overflow: hidden;
	}
	.mini i {
		flex: 1;
	}
	.matrix {
		display: grid;
		grid-template-columns: repeat(var(--n), var(--row));
		grid-auto-rows: var(--row);
		gap: var(--gap);
	}
	.matrix button {
		border: 0;
		border-radius: 3px;
		padding: 0;
		cursor: pointer;
		font-size: 9px;
		color: #1e1b4b;
		position: relative;
	}
	.matrix button.off {
		background: repeating-linear-gradient(
			-45deg,
			#f8fafc,
			#f8fafc 2px,
			#e2e8f0 2px,
			#e2e8f0 4px
		);
	}
	.matrix button.now {
		outline: 1px solid #7c3aed;
		z-index: 1;
	}
	.matrix button.pick {
		box-shadow: inset 0 0 0 1px #0f172a;
	}
	.state {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-top: 8px;
		color: #64748b;
		font-size: 12px;
	}
	.formula {
		margin: 6px 0 0;
		font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
		font-size: 11px;
		color: #64748b;
	}
	.probs {
		width: 230px;
	}
	.logit-hit {
		border: 0;
		background: transparent;
		padding: 0;
		cursor: pointer;
	}
	.readout {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 8px;
	}
	.from-cls {
		font-size: 11px;
		color: #92400e;
	}
	.bars {
		margin-top: 10px;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.bar-row {
		display: grid;
		grid-template-columns: 10px 64px 1fr 48px auto;
		gap: 6px;
		align-items: center;
		font-size: 12px;
		color: #334155;
	}
	.bar-row.picked {
		outline: 1px solid #0f172a;
		border-radius: 6px;
		padding: 2px;
	}
	.swatch {
		width: 8px;
		height: 8px;
		border-radius: 99px;
	}
	.track {
		height: 8px;
		background: #f1f5f9;
		border-radius: 99px;
		overflow: hidden;
	}
	.fill {
		height: 100%;
		border-radius: 99px;
	}
	.bar-row em {
		font-style: normal;
		font-size: 11px;
		color: #b91c1c;
	}
	.bar-row em.up {
		color: #047857;
	}
	.hint {
		margin: 4px 0 0;
		color: #94a3b8;
		font-size: 11px;
		line-height: 1.4;
	}
	.tip {
		position: fixed;
		z-index: 40;
		background: #fff;
		color: #475569;
		border: 1px solid #e2e8f0;
		border-radius: 6px;
		padding: 3px 7px;
		font-size: 12px;
		pointer-events: none;
		max-width: 240px;
	}
</style>
