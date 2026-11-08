/** Browser forward pass. The arithmetic matches train_vim.py. */

const EPS = 1e-5;

export function softmax(logits) {
	const max = Math.max(...logits);
	const exps = logits.map((value) => Math.exp(value - max));
	const sum = exps.reduce((total, value) => total + value, 0);
	return exps.map((value) => value / sum);
}

export function truncate(probs, sampling) {
	const next = probs.slice();
	const order = next.map((_, index) => index).sort((a, b) => next[b] - next[a]);
	if (sampling.type === "top-k") {
		const k = Math.max(1, Math.round(sampling.value));
		order.slice(k).forEach((index) => {
			next[index] = 0;
		});
	} else {
		const keep = new Set();
		let cum = 0;
		for (const index of order) {
			keep.add(index);
			cum += next[index];
			if (cum >= sampling.value) break;
		}
		next.forEach((_, index) => {
			if (!keep.has(index)) next[index] = 0;
		});
	}
	const sum = next.reduce((total, value) => total + value, 0) || 1;
	return next.map((value) => value / sum);
}

export function sampleIndex(probs, random = Math.random) {
	let cursor = random();
	let cum = 0;
	for (let index = 0; index < probs.length; index += 1) {
		cum += probs[index];
		if (cursor <= cum) return index;
	}
	return probs.length - 1;
}

function linear(vector, weight, bias) {
	return weight.map((row, index) => {
		let sum = bias ? bias[index] : 0;
		for (let i = 0; i < vector.length; i += 1) sum += vector[i] * row[i];
		return sum;
	});
}

function layerNorm(vector, weight, bias) {
	let mean = 0;
	for (const value of vector) mean += value;
	mean /= vector.length;
	let variance = 0;
	for (const value of vector) variance += (value - mean) ** 2;
	variance /= vector.length;
	const inv = 1 / Math.sqrt(variance + EPS);
	return vector.map((value, index) => (value - mean) * inv * weight[index] + bias[index]);
}

function siluValue(value) {
	if (value >= 0) return value / (1 + Math.exp(-value));
	const exp = Math.exp(value);
	return (value * exp) / (1 + exp);
}

function silu(vector) {
	return vector.map(siluValue);
}

function geluValue(value) {
	const inner = 0.7978845608028654 * (value + 0.044715 * value * value * value);
	return 0.5 * value * (1 + Math.tanh(inner));
}

function sigmoidValue(value) {
	if (value >= 0) return 1 / (1 + Math.exp(-value));
	const exp = Math.exp(value);
	return exp / (1 + exp);
}

function applyPointwise(sequence, fn) {
	return sequence.map((row) => row.map(fn));
}

function gateSequence(z, kind) {
	if (kind === "none") return z.map((row) => row.map(() => 1));
	if (kind === "relu") return applyPointwise(z, (value) => Math.max(0, value));
	if (kind === "gelu") return applyPointwise(z, geluValue);
	if (kind === "sigmoid") return applyPointwise(z, sigmoidValue);
	if (kind === "tanh") return applyPointwise(z, Math.tanh);
	return z.map(silu);
}

function scanActivation(kind) {
	if (kind === "relu") return (value) => Math.max(0, value);
	if (kind === "gelu") return geluValue;
	if (kind === "sigmoid") return sigmoidValue;
	if (kind === "tanh") return Math.tanh;
	if (kind === "none") return (value) => value;
	return siluValue;
}

function softplus(value) {
	if (value > 20) return value;
	if (value < -20) return Math.exp(value);
	return Math.log1p(Math.exp(value));
}

function add(left, right) {
	return left.map((value, index) => value + right[index]);
}

function mul(left, right) {
	return left.map((value, index) => value * right[index]);
}

function causalConv(sequence, weight, bias) {
	const kernel = weight[0].length;
	const out = [];
	for (let t = 0; t < sequence.length; t += 1) {
		const acc = bias.slice();
		for (let tap = 0; tap < kernel; tap += 1) {
			const src = t - (kernel - 1 - tap);
			if (src < 0) continue;
			for (let d = 0; d < acc.length; d += 1) acc[d] += weight[d][tap] * sequence[src][d];
		}
		out.push(acc);
	}
	return out;
}

function scan(x, dt, b, c, a, dSkip) {
	const inner = x[0].length;
	const state = a[0].length;
	const hidden = Array.from({ length: inner }, () => new Array(state).fill(0));
	const y = [];
	const states = [];
	for (let t = 0; t < x.length; t += 1) {
		const row = new Array(inner);
		const snapshot = [];
		for (let d = 0; d < inner; d += 1) {
			const next = new Array(state);
			let value = dSkip[d] * x[t][d];
			for (let n = 0; n < state; n += 1) {
				const aBar = Math.exp(dt[t][d] * a[d][n]);
				const bBar = dt[t][d] * b[t][n];
				next[n] = aBar * hidden[d][n] + bBar * x[t][d];
				value += c[t][n] * next[n];
			}
			hidden[d] = next;
			snapshot.push(next);
			row[d] = value;
		}
		states.push(snapshot);
		y.push(row);
	}
	return { y, states };
}

function influence(x, dt, b, c, a) {
	const length = x.length;
	const inner = x[0].length;
	const state = a[0].length;
	const matrix = Array.from({ length }, () => new Array(length).fill(0));
	for (let d = 0; d < inner; d += 1) {
		for (let i = 0; i < length; i += 1) {
			const carry = new Array(state);
			for (let n = 0; n < state; n += 1) carry[n] = dt[i][d] * b[i][n] * x[i][d];
			for (let t = i; t < length; t += 1) {
				if (t > i) {
					for (let n = 0; n < state; n += 1) carry[n] *= Math.exp(dt[t][d] * a[d][n]);
				}
				let contrib = 0;
				for (let n = 0; n < state; n += 1) contrib += c[t][n] * carry[n];
				matrix[t][i] += Math.abs(contrib);
			}
		}
	}
	for (let t = 0; t < length; t += 1) {
		let sum = 0;
		for (let i = 0; i < length; i += 1) sum += matrix[t][i];
		if (sum > 0) {
			for (let i = 0; i < length; i += 1) matrix[t][i] /= sum;
		}
	}
	return matrix;
}

function stateRms(states) {
	return states.map((snapshot) => {
		const state = snapshot[0].length;
		const inner = snapshot.length;
		const out = new Array(state).fill(0);
		for (let d = 0; d < inner; d += 1) {
			for (let n = 0; n < state; n += 1) out[n] += snapshot[d][n] ** 2;
		}
		return out.map((value) => Math.sqrt(value / inner));
	});
}

function flipMatrix(matrix) {
	const length = matrix.length;
	const out = Array.from({ length }, () => new Array(length).fill(0));
	for (let t = 0; t < length; t += 1) {
		for (let i = 0; i < length; i += 1) out[length - 1 - t][length - 1 - i] = matrix[t][i];
	}
	return out;
}

function scaledA(matrix, scale) {
	if (scale == null || scale === 1) return matrix;
	return matrix.map((row) => row.map((value) => value * scale));
}

function runDirection(x, direction, options = {}) {
	const conv = options.conv === false ? x : causalConv(x, direction.conv_w, direction.conv_b);
	const activated = applyPointwise(conv, scanActivation(options.scanAct));
	const rank = direction.dt_weight[0].length;
	const matrixA = scaledA(direction.A, options.aScale);
	const state = matrixA[0].length;
	const deltaRank = [];
	const delta = [];
	const b = [];
	const c = [];
	for (const row of activated) {
		const projected = linear(row, direction.x_weight, null);
		const low = projected.slice(0, rank);
		deltaRank.push(low);
		const step = options.deltaScale == null ? 1 : options.deltaScale;
		delta.push(linear(low, direction.dt_weight, direction.dt_bias).map((value) => softplus(value) * step));
		b.push(projected.slice(rank, rank + state));
		c.push(projected.slice(rank + state));
	}
	const scanned = scan(activated, delta, b, c, matrixA, direction.D);
	return {
		conv: activated,
		deltaRank,
		delta,
		b,
		c,
		y: scanned.y,
		state: stateRms(scanned.states),
		influence: influence(activated, delta, b, c, matrixA),
		A: matrixA
	};
}

function patchify(image, patch) {
	const grid = image[0].length / patch;
	const patches = [];
	const thumbs = [];
	for (let row = 0; row < grid; row += 1) {
		for (let col = 0; col < grid; col += 1) {
			const flat = [];
			const thumb = [[], [], []];
			for (let channel = 0; channel < 3; channel += 1) {
				for (let y = 0; y < patch; y += 1) {
					const line = [];
					for (let x = 0; x < patch; x += 1) {
						const value = image[channel][row * patch + y][col * patch + x];
						flat.push(value);
						line.push(value);
					}
					thumb[channel].push(line);
				}
			}
			patches.push(flat);
			thumbs.push(thumb);
		}
	}
	return { patches, thumbs };
}

export function forward(weights, image, options = {}) {
	const keep = options.keep || "both";
	const meta = weights.meta;
	const { patches, thumbs } = patchify(image, meta.patch);
	const embedded = patches.map((patch) => linear(patch, weights.patch_weight, weights.patch_bias));
	const sequence = [
		...embedded.slice(0, meta.clsAt),
		weights.cls.slice(),
		...embedded.slice(meta.clsAt)
	];
	const thumbSeq = [...thumbs.slice(0, meta.clsAt), null, ...thumbs.slice(meta.clsAt)];
	const grid = meta.image / meta.patch;
	const labels = [];
	for (let row = 0; row < grid; row += 1) {
		for (let col = 0; col < grid; col += 1) labels.push(`${row},${col}`);
	}
	labels.splice(meta.clsAt, 0, "CLS");
	const tokens = sequence.map((vector, index) => ({
		label: labels[index],
		cls: index === meta.clsAt,
		embedding: vector,
		position: weights.pos[index].slice(),
		sum: add(vector, weights.pos[index]),
		thumb: thumbSeq[index]
	}));

	let stream = tokens.map((token) => token.sum.slice());
	const blockList =
		options.layers == null ? weights.blocks : weights.blocks.slice(0, options.layers);
	const blocks = blockList.map((block) => {
		const input = stream.map((row) => row.slice());
		const normed = input.map((row) => layerNorm(row, block.ln_weight, block.ln_bias));
		const projected = normed.map((row) => linear(row, block.in_weight, block.in_bias));
		const x = projected.map((row) => row.slice(0, meta.inner));
		const z = projected.map((row) => row.slice(meta.inner));
		const forwardScan = runDirection(x, block.forward, options);
		const backwardRaw = runDirection(x.slice().reverse(), block.backward, options);
		const backward = {
			conv: backwardRaw.conv.slice().reverse(),
			deltaRank: backwardRaw.deltaRank.slice().reverse(),
			delta: backwardRaw.delta.slice().reverse(),
			b: backwardRaw.b.slice().reverse(),
			c: backwardRaw.c.slice().reverse(),
			y: backwardRaw.y.slice().reverse(),
			state: backwardRaw.state.slice().reverse(),
			influence: flipMatrix(backwardRaw.influence)
		};
		const merged = forwardScan.y.map((row, index) => {
			if (keep === "forward") return row.slice();
			if (keep === "backward") return backward.y[index].slice();
			return add(row, backward.y[index]);
		});
		const gate = gateSequence(z, options.gate);
		const gated = merged.map((row, index) => mul(row, gate[index]));
		const out = gated.map((row) => linear(row, block.out_weight, block.out_bias));
		const output =
			options.residual === false ? out.map((row) => row.slice()) : input.map((row, index) => add(row, out[index]));
		stream = output;
		return { input, normed, x, z, gate, forward: forwardScan, backward, merged, gated, out, output };
	});

	const finalNorm = stream.map((row) =>
		layerNorm(row, weights.final_ln_weight, weights.final_ln_bias)
	);
	const cls = finalNorm[meta.clsAt];
	const logits = linear(cls, weights.head_weight, weights.head_bias);
	return { tokens, blocks, finalNorm, cls, logits };
}

export function tokenFromPatch(patchIndex, clsAt) {
	return patchIndex < clsAt ? patchIndex : patchIndex + 1;
}

export function patchFromToken(tokenIndex, clsAt) {
	if (tokenIndex === clsAt) return -1;
	return tokenIndex < clsAt ? tokenIndex : tokenIndex - 1;
}

export function patchesFromRow(row, clsAt) {
	const values = [];
	for (let index = 0; index < row.length; index += 1) {
		if (index !== clsAt) values.push(row[index]);
	}
	return values;
}

function filledDelta(delta, value) {
	return delta.map((row) => row.map(() => value));
}

function packRead(matrix, clsAt) {
	return {
		patches: patchesFromRow(matrix[clsAt], clsAt),
		self: matrix[clsAt][clsAt]
	};
}

function traceChannel(scan, a, channel, state) {
	let hidden = 0;
	let integrator = 0;
	let forget = 0;
	return scan.conv.map((row, t) => {
		const delta = scan.delta[t][channel];
		const input = row[channel];
		const decay = Math.exp(delta * a[channel][state]);
		const write = delta * scan.b[t][state] * input;
		const previous = hidden;
		hidden = decay * previous + write;
		integrator += write;
		forget = 0.5 * forget + input;
		return { delta, input, decay, write, previous, hidden, integrator, forget };
	});
}

export function recurrenceTrace(scan, a) {
	let chosen = null;
	let best = -1;
	for (let channel = 0; channel < scan.conv[0].length; channel += 1) {
		for (let state = 0; state < a[0].length; state += 1) {
			const steps = traceChannel(scan, a, channel, state);
			const values = steps.map((step) => step.hidden);
			const range = Math.max(...values) - Math.min(...values);
			if (range > best) {
				best = range;
				chosen = steps;
			}
		}
	}
	return chosen;
}

export function classReads(scan, a, clsAt) {
	const learned = influence(scan.conv, scan.delta, scan.b, scan.c, a);
	const lasting = influence(scan.conv, filledDelta(scan.delta, 0.05), scan.b, scan.c, a);
	const forgetful = influence(scan.conv, filledDelta(scan.delta, 2), scan.b, scan.c, a);
	return {
		learned: packRead(learned, clsAt),
		lasting: packRead(lasting, clsAt),
		forgetful: packRead(forgetful, clsAt)
	};
}

function finite(value) {
	if (Number.isFinite(value)) return value;
	if (value > 0) return 1e6;
	if (value < 0) return -1e6;
	return 0;
}

export function classify(weights, image, options = {}) {
	const result = forward(weights, image, options);
	const logits = result.logits.map(finite);
	const probs = softmax(logits);
	const best = probs.indexOf(Math.max(...probs));
	const last = result.blocks[result.blocks.length - 1];
	const state = last.forward.state[weights.meta.clsAt];
	const stateSize = Math.sqrt(state.reduce((sum, value) => sum + value * value, 0) / state.length);
	return { logits, probs, best, stateSize, finite: result.logits.every(Number.isFinite) };
}

export function earlyGradient(weights, image, options = {}) {
	const cleanRun = forward(weights, image, options);
	const clean = classify(weights, image, options);
	const eps = 0.05;
	const bumped = image.map((channel) => channel.map((row) => row.slice()));
	for (let channel = 0; channel < 3; channel += 1) {
		for (let y = 0; y < weights.meta.patch; y += 1) {
			for (let x = 0; x < weights.meta.patch; x += 1) bumped[channel][y][x] += eps;
		}
	}
	const next = classify(weights, bumped, options);
	const raw = (next.logits[clean.best] - clean.logits[clean.best]) / eps;
	const last = cleanRun.blocks[cleanRun.blocks.length - 1].forward;
	const carry = decayProduct(last, last.A, 0, weights.meta.clsAt);
	return {
		...clean,
		carry,
		slope: Number.isFinite(raw) ? raw : raw > 0 ? 1e6 : -1e6
	};
}

export function decayProduct(scan, a, from, to) {
	const channels = scan.delta[0].length;
	const states = a[0].length;
	let largest = 0;
	for (let channel = 0; channel < channels; channel += 1) {
		for (let state = 0; state < states; state += 1) {
			let product = 1;
			for (let t = from + 1; t <= to; t += 1) {
				product *= Math.exp(scan.delta[t][channel] * a[channel][state]);
			}
			if (Math.abs(product) > Math.abs(largest)) largest = product;
		}
	}
	return largest;
}

export function mixedPatches(forwardRow, backwardRow, clsAt) {
	const forward = patchesFromRow(forwardRow, clsAt);
	const backward = patchesFromRow(backwardRow, clsAt);
	const mixed = forward.map((value, index) => value + backward[index]);
	const sum = mixed.reduce((total, value) => total + value, 0) || 1;
	return mixed.map((value) => value / sum);
}
