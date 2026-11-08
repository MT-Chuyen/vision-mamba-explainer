<script>
	import { PAGES } from "../lib/pages.js";

	export let page = 0;
	export let open = true;

	$: current = PAGES[page];
</script>

<aside class="book" class:closed={!open}>
	<div class="book-top">
		<button type="button" class="text" on:click={() => (open = !open)}>
			{open ? "Hide textbook" : "Textbook"}
		</button>
		{#if open}
			<span>{page + 1} / {PAGES.length}</span>
		{/if}
	</div>
	{#if open}
		<article>
			<h2>{current.title}</h2>
			<div class="body">{@html current.html}</div>
		</article>
		<div class="nav">
			<button type="button" disabled={page === 0} on:click={() => (page -= 1)}>Back</button>
			<button type="button" disabled={page === PAGES.length - 1} on:click={() => (page += 1)}>
				Next
			</button>
		</div>
	{/if}
</aside>

<style>
	.book {
		border-left: 1px solid #e2e8f0;
		background: #fff;
		padding: 16px 16px 20px;
		display: flex;
		flex-direction: column;
		gap: 12px;
		min-height: 100%;
	}
	.book.closed {
		padding-bottom: 16px;
	}
	.book-top {
		display: flex;
		justify-content: space-between;
		align-items: center;
		color: #94a3b8;
		font-size: 12px;
	}
	.text {
		border: 0;
		background: transparent;
		color: #64748b;
		padding: 0;
		cursor: pointer;
		font: inherit;
	}
	h2 {
		margin: 0 0 8px;
		font-size: 18px;
		font-weight: 650;
		color: #0f172a;
	}
	.body {
		color: #334155;
		font-size: 14px;
		line-height: 1.5;
	}
	.body :global(p) {
		margin: 0 0 8px;
	}
	.body :global(ol),
	.body :global(ul) {
		margin: 0 0 8px;
		padding-left: 18px;
	}
	.body :global(pre) {
		background: #f8fafc;
		border: 1px solid #e2e8f0;
		border-radius: 8px;
		padding: 8px 10px;
		font-size: 12px;
		line-height: 1.45;
		overflow-x: auto;
	}
	.nav {
		display: flex;
		gap: 8px;
		margin-top: auto;
	}
	.nav button {
		flex: 1;
		border: 1px solid #cbd5e1;
		background: #fff;
		border-radius: 8px;
		padding: 8px 10px;
		cursor: pointer;
		font: inherit;
	}
	.nav button:disabled {
		opacity: 0.4;
		cursor: default;
	}
</style>
