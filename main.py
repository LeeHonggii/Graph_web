from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import json
import PDFgraph
from langchain_core.messages import HumanMessage

app = FastAPI()

# Static 파일과 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class ProcessingConfig(BaseModel):
    pdf_module: str
    chunk_type: str
    chunk_size: Optional[int] = None
    embedding_module: str
    vector_module: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    config: str = Form(...)
):
    try:
        import base64
        
        # config JSON 파싱
        config_dict = json.loads(config)
        config_obj = ProcessingConfig(**config_dict)
        
        # PDF 파일 내용 읽고 base64로 인코딩
        content = await file.read()
        content_base64 = base64.b64encode(content).decode('utf-8')
        
        # 파이프라인 설정
        pipeline_config = {
            'pdf_module': config_obj.pdf_module,
            'chunk_type': config_obj.chunk_type,
            'chunk_size': config_obj.chunk_size,
            'embedding_module': config_obj.embedding_module,
            'vector_module': config_obj.vector_module
        }
        
        # RAG 파이프라인 생성
        pipeline = PDFgraph.create_rag_pipeline(pipeline_config)
        
        # 초기 상태 생성
        initial_state = PDFgraph.GraphState(
            messages=[HumanMessage(content=content_base64)],
            pdf_content="",
            chunks=[],
            embeddings=[],
            texts=[],
            page_info=[],
            vector_store=None,
            context="",
            response=""
        )
        
        # 파이프라인 실행
        result = pipeline.invoke(initial_state)
        
        return JSONResponse(content={
            "status": "success",
            "message": "PDF processed successfully",
            "result": {
                "chunks_count": len(result["chunks"]) if "chunks" in result else 0,
                "embeddings_count": len(result["embeddings"]) if "embeddings" in result else 0,
                "vector_store_status": result.get("vector_store", {}).get("status", "unknown")
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
def search_in_chroma(query: str, n_results: int = 3, embedding_type: str = "openai"):
    """Chroma DB에서 검색"""
    import chromadb
    import os
    
    # Chroma 클라이언트 초기화
    persist_directory = os.path.join(os.getcwd(), "db")
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        # 컬렉션 가져오기 (임베딩 타입에 따른 구분)
        collection_name = f"pdf_documents_{embedding_type}"
        collection = client.get_collection(collection_name)
        
        # 임베딩 타입에 따라 다른 처리
        if embedding_type == "openai":
            from openai import OpenAI
            openai_client = OpenAI()
            query_response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_embedding = query_response.data[0].embedding
        else:  # sentence-transformer
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            query_embedding = model.encode(query, convert_to_tensor=False).tolist()
        
        # 검색 실행
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # 결과 정리
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return search_results
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []
    
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        query = data.get('message', '')
        embedding_type = data.get('embedding_type', 'openai')  # 기본값은 openai
        
        if not query:
            return JSONResponse(content={
                "status": "error",
                "message": "Query is empty"
            })
        
        # embedding_type 전달
        search_results = search_in_chroma(query, embedding_type=embedding_type)
        
        if not search_results:
            return JSONResponse(content={
                "status": "success",
                "message": "죄송합니다. 관련된 내용을 찾을 수 없습니다."
            })
        
        context_texts = [f"페이지 {result['metadata']['page']}: {result['text']}" 
                        for result in search_results]
        
        response_text = f"검색 결과:\n\n" + "\n\n".join(context_texts)
        
        return JSONResponse(content={
            "status": "success",
            "message": response_text
        })
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
    
@app.get("/available-modules")
async def get_available_modules():
    return {
        "pdf_modules": ["pdfplumber", "pypdf", "pdfminer"],
        "chunk_types": ["sentence", "token"],
        "embedding_modules": ["openai", "sentence-transformer"],
        "vector_modules": ["chroma", "pinecone"]
    }