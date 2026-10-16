x"""
Standalone diagnostic: hits the NDTV/Gadgets360 reviews endpoint directly,
with no app caching, no logging config, nothing in the way. Run this from
your backend's virtualenv/conda env:

    python diagnose_reviews.py

Whatever it prints IS the real problem.
"""

import asyncio
import httpx

BASE_URL = "https://search.ndtv.com/news/json/client_key/ndtv-tech-e2a4508398a85f109cbbd47c181cb7d1"

PARAMS = {
    "blog_id": 9,
    "order_by": "published",
    "direction": "DESC",
    "extra_params": "category,category_slug,content_type,short_headline,authored,by_line,tags,keywords,categories,edited_by,written_by,by_line",
    "pagenumber": 1,
    "pagesize": 3,
    "categories": "",
    "content_type": "reviews",
}


async def main():
    url = f"{BASE_URL}/"
    print(f"Requesting: {url}")
    print(f"Params: {PARAMS}\n")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, params=PARAMS)
            print(f"Status code: {response.status_code}")
            print(f"Final URL: {response.url}")
            print(f"Response headers: {dict(response.headers)}\n")
            response.raise_for_status()
            data = response.json()
            print(f"Got {len(data.get('results', []))} results.")
            for item in data.get("results", [])[:3]:
                print(" -", item.get("title"))
    except httpx.HTTPStatusError as exc:
        print(f"HTTP ERROR: {exc.response.status_code}")
        print("Body:", exc.response.text[:500])
    except httpx.RequestError as exc:
        print(f"REQUEST ERROR (network/DNS/SSL/timeout): {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
