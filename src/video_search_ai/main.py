import os 
import argparse
from dotenv import load_dotenv

load_dotenv()

from pipeline import process_video
from video_db import AudioDatabase, VisualDatabase
from agent_call import agent_response
from playback import play_segment


def build_parser():
    parser = argparse.ArgumentParser(description = "Video Search AI parser")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")
    return parser, subparsers

def main():
    parser, subparser = build_parser()

    index_parser = subparser.add_parser("index", help = "Index a video file (speech/visual fallback)")
    index_parser.add_argument("video", type=str, help = "Path to the video file")

    search_parser = subparser.add_parser("search", help = "Search through indexed video segments")
    search_parser.add_argument("query", type=str, help = "Natural language query string")
    search_parser.add_argument("-n", "--results", type=int, default=3, help = "Number of results to return")

    args = parser.parse_args()

    if args.command == "index":
        video_path = args.video 
        print(f"Target file: {args.video}")
        process_video(video_path)

    elif args.command == "search":
        audio_db = AudioDatabase()
        visual_db = VisualDatabase()

        audio_result = audio_db.search(args.query, n_res = args.results)
        visual_result = visual_db.search(args.query, n_res = args.results)
        print(f"\nTop Search Results for: '{args.query}'\n" + "-"*55)

        audio_docs = (audio_result.get('documents') or [[]])[0]
        audio_meta = (audio_result.get('metadatas') or [[]])[0]
        visual_docs = (visual_result.get('documents') or [[]])[0]
        visual_meta = (visual_result.get('metadatas') or [[]])[0]
        
        combined = (
            [(doc, meta, "audio") for doc, meta in zip(audio_docs, audio_meta)]
            + [(doc, meta, "visual") for doc, meta in zip(visual_docs, visual_meta)]
        )

        print(f"\nTop Search Results for: '{args.query}'\n" + "-" * 55)
        if not combined:
            print("No matching video segments found.")
            return

        for doc, meta, modality in combined:
            video_name = meta.get('video', 'Unknown')
            start = meta.get('start', 0.0)
            end = meta.get('end', 0.0)
            print(f"[{start:.1f}s - {end:.1f}s] | File: {video_name} | {modality}")
            print(f"  -> {doc}\n")

        response = agent_response(
            query = args.query, 
            retrieved_docs = [doc for doc, _, _ in combined], 
            metadata = [{**meta, "modality": modality} for _, meta, modality in combined]
        )
        print(response["answer"])

        video, start, end = response.get("video"), response.get("start"), response.get("end")
        if video and start is not None:
            play_segment(video, start, end)


if __name__ == "__main__":
    main()