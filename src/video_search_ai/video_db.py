import numpy as np 
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch 
import cv2 

import chromadb
from sentence_transformers import SentenceTransformer

class AudioDatabase:
    """
    Stores the embedding vector of the detected audio/transcript from the video
    Target storage : text vectors (MiniLM-L6-v2)
    """
    def __init__(self, db_path: str = "./chroma-db-audio"):
        self.client = chromadb.PersistentClient(path = db_path)
        self.collection = self.client.get_or_create_collection("video_search")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_segment(self, video_name : str, start : float, end : float, text : str):
        vector = self.encoder.encode(text).tolist()
        doc_id = f"{video_name}_{start}_{end}"
        self.collection.add(
            documents=[text],
            embeddings=[vector],
            metadatas=[{"video": video_name, "start": start, "end": end}],
            ids=[doc_id]
        )

    def search(self, query: str, n_res: int = 3):
        query_vector = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings = [query_vector],
            n_results = n_res
        )
        return results

class VisualDatabase:
    """
    Stores the embedding vectors of visual keyframes captured from video
    Target storage : image(frame) vectors (CLIP)
    """
    def __init__(self, db_path: str = "./chroma-db-frames", collection_name:str = "visual_frames", model:str = "openai/clip-vit-base-patch32"):
        self.client = chromadb.PersistentClient(db_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model, use_safetensors=True).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model)


    def _embed_frame(self, frame_bgr : np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        inputs = self.processor(images = image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        feats = features / features.norm(p=2, dim=-1, keepdim=True) 
        return feats.squeeze(0).cpu().tolist()


    def _embed_text(self, text):
        inputs = self.processor(text = [text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.squeeze(0).cpu().tolist()


    def add_keyframe(self, video_name: str, start: float, end: float, frame_bgr: np.ndarray, frame_path: str):
        frame_vector = self._embed_frame(frame_bgr)
        doc_id = f"{video_name}_{start:.1f}_{end:.1f}_visual"

        self.collection.upsert(
            documents = [f"[Visual keyframe] frame at {start:.1f}s of {video_name}"],
            embeddings = [frame_vector],
            metadatas = [{
                "video" : video_name, 
                "start" : start, 
                "end" : end, 
                "modality" : "visual",
                "frame_path" : frame_path
            }],
            ids = [doc_id]
        )
    
    def search(self, query: str, n_res:int = 2):
        query_vector = self._embed_text(query)
        return self.collection.query(query_embeddings=[query_vector], n_results=n_res)