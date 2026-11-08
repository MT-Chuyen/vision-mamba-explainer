export function shade(value, values, hue) {
	let max = 1e-6;
	for (const item of values) max = Math.max(max, Math.abs(item));
	const t = Math.max(-1, Math.min(1, value / max));
	const light = 94 - ((t + 1) / 2) * 54;
	return `hsl(${hue} 64% ${light}%)`;
}

export function heat(value) {
	const t = Math.max(0, Math.min(1, value));
	return `hsl(262 70% ${96 - t * 50}%)`;
}

export const CLASS_COLORS = ["#e85d4c", "#3b6fe0", "#1f9d55", "#e3b341"];

export function paintImage(canvas, image, scale = 7) {
	const size = image[0].length;
	canvas.width = size * scale;
	canvas.height = size * scale;
	const ctx = canvas.getContext("2d");
	for (let y = 0; y < size; y += 1) {
		for (let x = 0; x < size; x += 1) {
			const r = Math.round(image[0][y][x] * 255);
			const g = Math.round(image[1][y][x] * 255);
			const b = Math.round(image[2][y][x] * 255);
			ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
			ctx.fillRect(x * scale, y * scale, scale, scale);
		}
	}
}

export function paintThumb(node, thumb) {
	const draw = (value) => {
		if (!value) return;
		const size = value[0].length;
		node.width = size;
		node.height = size;
		const ctx = node.getContext("2d");
		const pixels = ctx.createImageData(size, size);
		for (let y = 0; y < size; y += 1) {
			for (let x = 0; x < size; x += 1) {
				const offset = (y * size + x) * 4;
				pixels.data[offset] = Math.round(value[0][y][x] * 255);
				pixels.data[offset + 1] = Math.round(value[1][y][x] * 255);
				pixels.data[offset + 2] = Math.round(value[2][y][x] * 255);
				pixels.data[offset + 3] = 255;
			}
		}
		ctx.putImageData(pixels, 0, 0);
	};
	draw(thumb);
	return { update: draw };
}
