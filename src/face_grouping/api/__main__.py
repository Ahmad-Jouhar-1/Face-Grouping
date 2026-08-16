"""Run the optional FastAPI adapter with Uvicorn."""
import os

import uvicorn


def main() -> None:
    host = os.environ.get("FACE_GROUPING_API_HOST", "127.0.0.1")
    port = int(os.environ.get("FACE_GROUPING_API_PORT", "8001"))
    uvicorn.run("face_grouping.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
