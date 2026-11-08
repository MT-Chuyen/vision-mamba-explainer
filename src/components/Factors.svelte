<script>
	import weights from "../lib/weights.json";
	import examples from "../lib/examples.json";
	import { classify, earlyGradient, forward } from "../lib/vim.js";
	import { CLASS_COLORS } from "../lib/color.js";

	export let image;
	export let classes = [];

	const FUNCTIONS = [
		["SiLU", "silu"],
		["GELU", "gelu"],
		["ReLU", "relu"],
		["sigmoid", "sigmoid"],
		["tanh", "tanh"],
		["none", "none"]
	];
	const SHAPES = examples.examples;

	function sigmoid(value) {
		if (value >= 0) return 1 / (1 + Math.exp(-value));
		const exp = Math.exp(value);
		return exp / (1 + exp);
	}

	function gelu(value) {
		const inner = 0.7978845608028654 * (value + 0.044715 * value * value * value);
		return 0.5 * value * (1 + Math.tanh(inner));
	}

	function gateDerivative(kind, value) {
		if (kind === "none") return 0;
		if (kind === "relu") return value > 0 ? 1 : 0;
		if (kind === "sigmoid") {
			const s = sigmoid(value);
			return s * (1 - s);
		}
		if (kind === "tanh") {
			const t = Math.tanh(value);
			return 1 - t * t;
		}
		if (kind === "gelu") return (gelu(value + 1e-3) - gelu(value - 1e-3)) / 2e-3;
		const s = sigmoid(value);
		return s * (1 + value * (1 - s));
	}

	function chipsFor(options) {
		return SHAPES.map((example) => {
			const result = classify(weights, example.image, options);
			return {
				label: example.label,
				predicted: weights.meta.classes[result.best],
				ok: result.best === example.classIndex,
				color: CLASS_COLORS[result.best]
			};
		});
	}

	function board(makeOptions) {
		return FUNCTIONS.map(([label, id]) => ({
			label,
			id,
			chips: chipsFor(makeOptions(id))
		}));
	}

	const gateBoard = board((id) => ({ gate: id }));
	const scanBoard = board((id) => ({ scanAct: id }));
	const factors = [
		["Trained", {}],
		["No residual", { residual: false }],
		["1 block", { layers: 1 }],
		["2 blocks", { layers: 2 }],
		["No convolution", { conv: false }]
	].map(([label, options]) => ({ label, cells: chipsFor(options) }));

	let gateId = "silu";
	let scanId = "silu";
	let step = 1;
	let grow = false;

	function quiet(grads) {
		return grads.filter((value) => Math.abs(value) < 0.05).length;
	}

	function stopped(grads) {
		return grads.filter((value) => Math.abs(value) < 1e-4).length;
	}

	function carryText(value) {
		if (!Number.isFinite(value)) return "overflow";
		if (value >= 10) return value >= 1000 ? `×${value.toExponential(1)}` : `×${value.toFixed(0)}`;
		return `${(value * 100).toFixed(0)}% left`;
	}

	function stateText(value) {
		if (!Number.isFinite(value)) return "overflow";
		return value >= 1000 ? value.toExponential(1) : value.toExponential(1);
	}

	$: gatePick = gateBoard.find((entry) => entry.id === gateId);
	$: scanPick = scanBoard.find((entry) => entry.id === scanId);
	$: gateNow = image ? classify(weights, image, { gate: gateId }) : null;
	$: scanNow = image ? classify(weights, image, { scanAct: scanId }) : null;
	$: shock = image
		? earlyGradient(weights, image, { deltaScale: Number(step), aScale: grow ? -1 : 1 })
		: null;
	$: shockChips = chipsFor({ deltaScale: Number(step), aScale: grow ? -1 : 1 });
	$: zRow = image ? forward(weights, image).blocks[0].z[weights.meta.clsAt] : [];
	$: siluGrad = zRow.map((value) => gateDerivative("silu", value));
	$: pickedGrad = zRow.map((value) => gateDerivative(gateId, value));
	$: gradPeak = Math.max(...siluGrad.map(Math.abs), ...pickedGrad.map(Math.abs), 1e-6);
</script>

<section class="factors" data-col="factors">
	<h2>Change one piece</h2>
	<div class="grid">
		<article>
			<h3>Swap the gate</h3>
			<p>
				The gate multiplies the scan, channel by channel. Pick another function. The bars are this picture. The four names are the four shapes, rerun with that function and nothing else changed.
			</p>
			<div class="choices">
				{#each FUNCTIONS as [label, id]}
					<button type="button" class:on={gateId === id} on:click={() => (gateId = id)}>{label}</button>
				{/each}
			</div>
			{#if gateNow}
				<div class="bars">
					{#each gateNow.probs as probability, index}
						<div class="mini">
							<span>{classes[index]}</span>
							<div><b style:width={`${probability * 100}%`} style:background={CLASS_COLORS[index]}></b></div>
						</div>
					{/each}
				</div>
			{/if}
			<div class="chips">
				{#each gatePick.chips as chip}
					<span class:bad={!chip.ok} style:color={chip.color}>
						{chip.ok ? chip.label : `${chip.label} → ${chip.predicted}`}
					</span>
				{/each}
			</div>
			<p class="label">Gate gradient at the class token, this block. Height is |d(gate)/dz|.</p>
			<div class="strips" aria-hidden="true">
				<div>
					<span>SiLU</span>
					<div class="strip">
						{#each siluGrad as value}
							<i style:height={`${(Math.abs(value) / gradPeak) * 100}%`}></i>
						{/each}
					</div>
				</div>
				<div>
					<span>{gatePick.label}</span>
					<div class="strip alt">
						{#each pickedGrad as value}
							<i style:height={`${(Math.abs(value) / gradPeak) * 100}%`}></i>
						{/each}
					</div>
				</div>
			</div>
			<p class="note">
				{stopped(pickedGrad)} of {pickedGrad.length} channels have a gate gradient of 0 under {gatePick.label}.
				{quiet(siluGrad)} channels are under 0.05 with SiLU, and {quiet(pickedGrad)} with {gatePick.label}.
				ReLU's gradient is exactly 0 on every negative channel, so that channel stops learning. SiLU's gradient stays a small nonzero number there, and the gate can still close. Sigmoid never reaches 0, so it cannot close a channel. Removing the gate forces every scan channel through.
			</p>
		</article>

		<article>
			<h3>Swap the scan activation</h3>
			<p>
				After the width-3 convolution, one function is applied and then Δ, B, and C are built from that result. The trained function is SiLU. This is a different slot from the gate.
			</p>
			<div class="choices">
				{#each FUNCTIONS as [label, id]}
					<button type="button" class:on={scanId === id} on:click={() => (scanId = id)}>{label}</button>
				{/each}
			</div>
			{#if scanNow}
				<div class="bars">
					{#each scanNow.probs as probability, index}
						<div class="mini">
							<span>{classes[index]}</span>
							<div><b style:width={`${probability * 100}%`} style:background={CLASS_COLORS[index]}></b></div>
						</div>
					{/each}
				</div>
			{/if}
			<div class="chips">
				{#each scanPick.chips as chip}
					<span class:bad={!chip.ok} style:color={chip.color}>
						{chip.ok ? chip.label : `${chip.label} → ${chip.predicted}`}
					</span>
				{/each}
			</div>
			<p class="note">
				SiLU keeps the sign of each channel and is smooth at zero, so a negative conv output still contributes a little. ReLU deletes that half before Δ, B, and C are computed, and Square and Diagonal become Circle. Sigmoid folds every channel into (0, 1); the state grows about a hundred times and all four shapes become Diagonal. GELU stays with SiLU on these four. tanh and a linear pass also keep them, so the failure to avoid is a function that deletes the sign or saturates the size.
			</p>
		</article>

		<article>
			<h3>The sign of A, and the size of Δ</h3>
			<p>
				The gradient of a later state with respect to an earlier patch is the product of <code>exp(Δ A)</code> along the steps. A is stored as <code>−exp(A_log)</code>, so each factor is below 1.
			</p>
			<div class="choices">
				<button type="button" class:on={!grow} on:click={() => (grow = false)}>A negative, decay</button>
				<button type="button" class:on={grow} on:click={() => (grow = true)}>A positive, growth</button>
			</div>
			<label>
				Δ × {step}
				<input type="range" min="1" max="20" step="1" bind:value={step} />
			</label>
			{#if shock}
				<div class="meters">
					<div>
						<span>First patch at the class token</span>
						<b>{carryText(shock.carry)}</b>
						<div class="track">
							<i
								class:hot={grow}
								style:width={`${Math.min(100, Number.isFinite(shock.carry) ? (shock.carry >= 1 ? Math.log10(shock.carry + 1) * 28 : shock.carry * 100) : 100)}%`}
							></i>
						</div>
					</div>
					<div>
						<span>State size</span>
						<b>{stateText(shock.stateSize)}</b>
					</div>
					<div>
						<span>Corner still moves the winning logit</span>
						<b>{shock.finite ? shock.slope.toExponential(1) : "0"}</b>
					</div>
				</div>
				<div class="bars">
					{#each shock.probs as probability, index}
						<div class="mini">
							<span>{classes[index]}</span>
							<div><b style:width={`${probability * 100}%`} style:background={CLASS_COLORS[index]}></b></div>
						</div>
					{/each}
				</div>
				<div class="chips">
					{#each shockChips as chip}
						<span class:bad={!chip.ok} style:color={chip.color}>
							{chip.ok ? chip.label : `${chip.label} → ${chip.predicted}`}
						</span>
					{/each}
				</div>
			{/if}
			<p class="note">
				{#if shock && (!shock.finite || !Number.isFinite(shock.carry))}
					The state overflowed. The logits stop being numbers, the corner gradient drops to 0, and the four classes collapse to a guess.
				{:else if grow}
					With A positive the product grows on every step. At the learned Δ the growth is still mild, because Δ is about 0.02. Raise Δ and the same sign overflows the state, the corner gradient blows up and then dies, and the class changes.
				{:else}
					With A negative a larger Δ only forgets. The first patch's leftover falls, and the class can still hold because the residual adds the block input back around the scan.
				{/if}
			</p>
		</article>
	</div>

	<article class="wide">
		<h3>What moves the answer if you edit the stack</h3>
		<p>
			Each row keeps the trained weights and changes one structural choice. A red outline is a shape whose class changed. These are the edits that reach the final scores on this model.
		</p>
		<table>
			<thead>
				<tr>
					<th>Edit</th>
					{#each SHAPES as shape}
						<th>{shape.label}</th>
					{/each}
				</tr>
			</thead>
			<tbody>
				{#each factors as row}
					<tr>
						<th>{row.label}</th>
						{#each row.cells as cell}
							<td class:bad={!cell.ok} style:color={cell.color}>{cell.predicted}</td>
						{/each}
					</tr>
				{/each}
			</tbody>
		</table>
		<p class="note">
			The residual is the add-back around the scan. Drop it and Circle, Square, and Stripes become Diagonal. One block is not enough for Stripes or Diagonal; two blocks still confuse Diagonal with Square. The width-3 convolution is what Δ, B, and C look through. Remove it and Circle, Square, and Stripes become Diagonal. Depth, the residual, and that convolution are the three structural edits that move these scores. A smooth gate and a negative A are the two math edits that keep the gradient alive while you train.
		</p>
	</article>
</section>

<style>
	.factors {
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
		grid-template-columns: 1fr 1fr 1.05fr;
		gap: 10px;
	}
	article {
		border: 1px solid #e2e8f0;
		border-radius: 12px;
		padding: 12px;
		background: #f8fafc;
	}
	.wide {
		margin-top: 10px;
	}
	h3 {
		margin: 0 0 6px;
		font-size: 14px;
	}
	p,
	.note,
	.label {
		margin: 0 0 8px;
		color: #334155;
		font-size: 13px;
		line-height: 1.45;
	}
	.note,
	.label {
		color: #64748b;
		margin: 8px 0 0;
	}
	code {
		font-size: 12px;
	}
	.choices {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin-bottom: 8px;
	}
	button {
		border: 1px solid #cbd5e1;
		background: white;
		border-radius: 999px;
		padding: 3px 8px;
		font-size: 12px;
		color: #334155;
		cursor: pointer;
	}
	button.on {
		background: #172554;
		border-color: #172554;
		color: white;
	}
	.bars {
		margin-bottom: 6px;
	}
	.mini {
		display: grid;
		grid-template-columns: 58px 1fr;
		gap: 4px;
		align-items: center;
		font-size: 10px;
		color: #64748b;
		margin-top: 2px;
	}
	.mini div,
	.track {
		height: 6px;
		background: white;
		border-radius: 99px;
		overflow: hidden;
	}
	.mini b,
	.track i {
		display: block;
		height: 100%;
		border-radius: 99px;
		background: #7c3aed;
	}
	.track i.hot {
		background: #e11d48;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 8px;
	}
	.chips span {
		font-size: 11px;
		font-weight: 700;
	}
	.chips span.bad,
	td.bad {
		text-decoration: underline;
		text-decoration-color: #e11d48;
	}
	.strips {
		display: grid;
		gap: 4px;
		margin-top: 8px;
	}
	.strips > div {
		display: grid;
		grid-template-columns: 52px 1fr;
		gap: 6px;
		align-items: end;
	}
	.strips span {
		font-size: 11px;
		color: #64748b;
	}
	.strip {
		height: 28px;
		display: flex;
		align-items: flex-end;
		gap: 2px;
	}
	.strip i {
		flex: 1;
		background: #7c3aed;
		border-radius: 2px 2px 0 0;
		display: block;
	}
	.strip.alt i {
		background: #e11d48;
	}
	label {
		display: grid;
		gap: 4px;
		font-size: 12px;
		color: #334155;
		margin-bottom: 8px;
	}
	.meters {
		display: grid;
		gap: 6px;
		margin-bottom: 8px;
	}
	.meters span {
		display: block;
		font-size: 11px;
		color: #64748b;
	}
	.meters b {
		font-size: 13px;
		color: #0f172a;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 12px;
	}
	th,
	td {
		text-align: left;
		padding: 5px 8px;
		border-bottom: 1px solid #e2e8f0;
	}
	td {
		font-weight: 700;
	}
	@media (max-width: 1100px) {
		.grid {
			grid-template-columns: 1fr;
		}
	}
</style>
