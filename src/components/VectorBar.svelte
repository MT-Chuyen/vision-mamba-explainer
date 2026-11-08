<script>
	import { shade } from "../lib/color.js";

	export let values = [];
	export let hue = 217;
	export let anchor = "";
	export let tip = "";
	export let index = 0;
</script>

<div
	class="vector"
	data-anchor={anchor || undefined}
	data-index={index}
	data-tip={tip}
>
	{#each values as value}
		<i style:background={shade(value, values, hue)}></i>
	{/each}
</div>

<style>
	.vector {
		width: 16px;
		height: var(--row);
		display: flex;
		flex-direction: column;
		border-radius: 3px;
		overflow: hidden;
		flex: 0 0 auto;
		box-shadow: inset 0 0 0 1px rgb(15 23 42 / 8%);
	}
	i {
		flex: 1 1 0;
		display: block;
	}
</style>
