// Sample blog data - replace with actual posts
const posts = [
    {
        title: "How to Read an Options Chain",
        punchline: "Price, strike, expiry — three numbers that tell the whole story",
        author: "CFC Editorial",
        date: "2026-03-01",
        topic: "Stock Options",
        content: `An options chain lists every available contract for a given stock, organized by expiration date and strike price. Each row shows two sides: calls (the right to buy) and puts (the right to sell). The key columns to focus on are the bid/ask spread (the cost to enter a position), open interest (how many contracts are active — a proxy for liquidity), and implied volatility (IV), which reflects the market's expectation of future price movement. High IV means options are expensive; low IV means they're cheap. Before placing any trade, check that open interest is above a few hundred contracts so you can exit without getting stuck.`
    },
    {
        title: "Price-to-Earnings: The Most Misused Ratio in Investing",
        punchline: "A low P/E is not always a bargain — context is everything",
        author: "CFC Editorial",
        date: "2026-03-12",
        topic: "Stock Valuation",
        content: `The P/E ratio divides a stock's price by its earnings per share, giving a sense of what the market pays for each dollar of profit. A P/E of 15 is often called "cheap" and 30 "expensive," but those labels mean little without context. Growth stocks command high P/Es because investors are paying for future earnings, not today's. Cyclical stocks (energy, materials) can show deceptively low P/Es near the top of a business cycle, right before earnings collapse. The forward P/E — based on next year's estimated earnings — is usually more useful than the trailing figure. Compare P/E within the same industry, not across sectors, and always ask why a ratio is where it is.`
    },
    {
        title: "Covered Calls: Generating Income on Stocks You Already Own",
        punchline: "Turn your long position into a recurring income stream",
        author: "CFC Editorial",
        date: "2026-03-22",
        topic: "Stock Options",
        content: `A covered call is one of the simplest options strategies: you own 100 shares of a stock and sell one call contract against them, collecting the premium upfront. In exchange, you agree to sell your shares at the strike price if the stock rises above it by expiration. The premium lowers your effective cost basis and cushions small drawdowns. The trade-off is capped upside — if the stock surges past your strike, you miss the gain above it. Most traders sell calls 30–45 days out on stocks they're comfortable holding long-term, targeting strikes 5–10% above the current price. It works best in sideways or mildly bullish markets where the stock is unlikely to make a violent move.`
    },
    {
        title: "Market indices historical return",
        punchline: "Historical returns are skewed to the right",
        author: "Gonzalez",
        date: "2025-10-29",
        topic: "Market returns",
        content: `Historical calendar year performance for the S&P500, DJIA and NASDAQ indices
        are 
        S&P500 (1928 - 2024)
        Negative returns: 27%, 26/97 yrs
        Returns < 5%: 33%, 32/97 yrs
        Returns < 10%: 42%, 41/97 yrs
        DJIA (1928 - 2024)
        Negative returns: 32%, 31/97 yrs
        Returns < 5%: 42%, 41/97 yrs
        Returns < 10%: 51%, 49/97 yrs
        NASDAQ (1971 - 2024)
        Negative returns: 26%, 14/54 yrs
        Returns < 5%: 28%, 15/54 yrs
        Returns < 10%: 41%, 22/54 yrs
        `
    }
];
