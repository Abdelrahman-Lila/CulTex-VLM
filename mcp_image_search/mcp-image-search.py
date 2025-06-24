from mcp.server.fastmcp import FastMCP
from mcp.tools import serpapi_scrape_search

mcp = FastMCP("search")

@mcp.tool()
async def search_similar_images(image_url: str) -> str:
    try:
        params = {
            "engine": "google_reverse_image",
            "image_url": image_url,
            "device": "desktop",
            "noCache": True,
            "gl": "eg",
        }

        result = await serpapi_scrape_search(**params)

        if not result or "images_results" not in result:
            return "No similar images found or API error occurred."

        results = []
        for i, image in enumerate(result["images_results"][:5]):
            title = image.get("title", "No title")
            source = image.get("source", "Unknown source")
            thumbnail = image.get("thumbnail", "No thumbnail available")
            link = image.get("original", image.get("link", "No link available"))

            result_text = f"""Result {i+1}:
            Title: {title}
            Source: {source}
            Image URL: {link}
            Thumbnail: {thumbnail}
            """
            results.append(result_text)

        return "\n---\n".join(results) if results else "No similar images found."

    except Exception as e:
        return f"Error searching for similar images: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
