from typing import Dict, TypedDict, List, Callable, Any
from langgraph.graph import Graph, END,StateGraph
from langchain_core.messages import HumanMessage


# 상태 타입 정의
class GraphState(TypedDict):
    messages: list[HumanMessage]
    pdf_content: dict  # PDF 처리 결과 메타데이터
    chunks: dict      # 청크 처리 결과
    embeddings: list  # 임베딩 결과
    texts: list      # 청크 텍스트
    page_info: list  # 페이지 정보
    vector_store: any # 벡터 저장소 결과
    context: str
    response: str

# PDF 처리 함수들
def process_pdf_pdfplumber(content: bytes) -> dict:
    """PDFPlumber를 사용한 PDF 처리"""
    import pdfplumber
    from io import BytesIO
    
    metadata = {'pages': 0, 'metadata': {}}
    
    with pdfplumber.open(BytesIO(content)) as pdf:
        metadata['pages'] = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                metadata['metadata'][f'page_{i}'] = {
                    'content': text.strip(),
                    'chunks': []  # 청크는 나중에 추가됨
                }
    
    return metadata

def process_pdf_pypdf(content: bytes) -> dict:
    """PyPDF2를 사용한 PDF 처리"""
    from PyPDF2 import PdfReader
    from io import BytesIO
    
    metadata = {'pages': 0, 'metadata': {}}
    reader = PdfReader(BytesIO(content))
    
    metadata['pages'] = len(reader.pages)
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text:
            metadata['metadata'][f'page_{i}'] = {
                'content': text.strip(),
                'chunks': []
            }
    
    return metadata

def process_pdf_pdfminer(content: bytes) -> dict:
    """PDFMiner를 사용한 PDF 처리"""
    from pdfminer.high_level import extract_pages, extract_text
    from pdfminer.layout import LTTextContainer
    from io import BytesIO
    
    metadata = {'pages': 0, 'metadata': {}}
    text_content = {}
    
    # 페이지별로 텍스트 추출
    for i, page_layout in enumerate(extract_pages(BytesIO(content)), 1):
        texts = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                texts.append(element.get_text())
        
        if texts:
            metadata['metadata'][f'page_{i}'] = {
                'content': ''.join(texts).strip(),
                'chunks': []
            }
    
    metadata['pages'] = len(metadata['metadata'])
    return metadata

# 청크 처리 함수들
def chunk_by_sentence(metadata: dict) -> dict:
    """문장 단위로 청크 분할"""
    updated_metadata = {'pages': metadata['pages'], 'metadata': {}}
    chunk_id = 0
    
    for page_num, page_data in metadata['metadata'].items():
        content = page_data['content']
        # 엔터(\n) 단위로 분리
        sentences = [sent.strip() for sent in content.split('\n') if sent.strip()]
        
        # 각 줄을 개별 청크로 저장
        page_chunks = []
        for sentence in sentences:
            chunk_info = {
                'chunk_id': f'chunk_{chunk_id}',
                'content': sentence,
                'page': page_num
            }
            page_chunks.append(chunk_info)
            chunk_id += 1
        
        updated_metadata['metadata'][page_num] = {
            'content': content,
            'chunks': page_chunks
        }
    
    return updated_metadata
def chunk_by_token(metadata: dict, chunk_size: int) -> dict:
    """토큰 단위로 청크 분할"""
    import tiktoken
    
    enc = tiktoken.get_encoding("cl100k_base")
    updated_metadata = {'pages': metadata['pages'], 'metadata': {}}
    chunk_id = 0
    
    for page_num, page_data in metadata['metadata'].items():
        content = page_data['content']
        tokens = enc.encode(content)
        chunks = []
        
        # chunk_size 단위로 분할
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = enc.decode(chunk_tokens)
            
            if chunk_text.strip():  # 빈 청크 제외
                chunk_info = {
                    'chunk_id': f'chunk_{chunk_id}',
                    'content': chunk_text.strip(),
                    'page': page_num
                }
                chunks.append(chunk_info)
                chunk_id += 1
        
        updated_metadata['metadata'][page_num] = {
            'content': content,
            'chunks': chunks
        }
    
    return updated_metadata
# 임베딩 함수들
def embed_with_openai(metadata: dict) -> list:
    """OpenAI 임베딩"""
    from openai import OpenAI
    
    client = OpenAI()
    all_embeddings = []
    all_texts = []
    page_info = []
    
    # 모든 페이지의 청크들을 처리
    for page_num, page_data in metadata['metadata'].items():
        for chunk in page_data['chunks']:
            try:
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=chunk['content']  # 'content' 필드에서 텍스트 추출
                )
                all_embeddings.append(response.data[0].embedding)
                all_texts.append(chunk['content'])  # 텍스트만 저장
                page_info.append({'page': chunk['page'], 'chunk_id': chunk['chunk_id']})
            except Exception as e:
                print(f"Error on {page_num}, chunk {chunk['chunk_id']}: {str(e)}")
                continue
    
    return all_embeddings, all_texts, page_info

def embed_with_sentence_transformer(metadata: dict) -> list:
    """SentenceTransformer 임베딩"""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    # 모델 로딩 (전역 변수로 만들거나 캐싱하면 더 효율적)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    all_embeddings = []
    all_texts = []
    page_info = []
    
    # 모든 페이지의 청크들을 처리
    for page_num, page_data in metadata['metadata'].items():
        for chunk in page_data['chunks']:
            try:
                # 임베딩 생성
                embedding = model.encode(chunk['content'], convert_to_tensor=False)
                
                # NumPy 배열을 리스트로 변환
                embedding_list = embedding.tolist()
                
                all_embeddings.append(embedding_list)
                all_texts.append(chunk['content'])
                page_info.append({'page': chunk['page'], 'chunk_id': chunk['chunk_id']})
                
            except Exception as e:
                print(f"Error creating embedding for chunk {chunk['chunk_id']}: {str(e)}")
                continue
    
    if not all_embeddings:
        raise ValueError("No embeddings were created successfully")
        
    return all_embeddings, all_texts, page_info

def store_in_chroma(embeddings: list, texts: list, page_info: list, embedding_type: str = "openai", file_info: dict = None) -> dict:
    """Chroma에 벡터 저장"""
    import chromadb
    import os
    from datetime import datetime
    
    # Chroma 설정
    persist_directory = os.path.join(os.getcwd(), "db")
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    # 파일 이름 처리
    file_name = os.path.splitext(file_info.get("file_name", "unknown"))[0]
    file_name = ''.join(c if c.isalnum() else '_' for c in file_name)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    pdf_module = file_info.get("pdf_module", "unknown")
    chunk_type = file_info.get("chunk_type", "unknown")
    
    # chunk_size 처리
    chunk_size = "word" if chunk_type == "sentence" else str(file_info.get("chunk_size", "unknown"))
    
    # embedding_type 단순화
    embedding_type = "sbert" if embedding_type == "sentence-transformer" else "openai"
    
    collection_name = f"vec_{timestamp}_{file_name}_{pdf_module}_{chunk_type}_{chunk_size}_{embedding_type}"

    print(f"Creating collection: {collection_name}")
    
    # Chroma 클라이언트 초기화
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        # 컬렉션 생성
        collection = chroma_client.create_collection(name=collection_name)
        
        # 데이터 저장
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=page_info,
            ids=[info['chunk_id'] for info in page_info]
        )
        
        return {
            "status": "stored in chroma",
            "collection_name": collection_name,
            "total_chunks": len(texts)
        }
    
    except Exception as e:
        print(f"Error storing in Chroma: {str(e)}")
        raise e
def store_in_pinecone(embeddings: List[float], texts: List[str]) -> any:
    """Pinecone에 벡터 저장"""
    # Pinecone 저장 구현
    return {"status": "stored in pinecone"}

def create_rag_pipeline(config: Dict[str, str]) -> Graph:
    """설정에 따라 RAG 파이프라인 생성"""
    # PDF 처리 함수 선택
    pdf_functions = {
        'pdfplumber': process_pdf_pdfplumber,
        'pypdf': process_pdf_pypdf,
        'pdfminer': process_pdf_pdfminer
    }
    pdf_processor = pdf_functions[config['pdf_module']]
    
    # 청크 처리 함수 선택
    chunk_functions = {
        'sentence': chunk_by_sentence,
        'token': lambda x: chunk_by_token(x, config['chunk_size'])
    }
    chunk_processor = chunk_functions[config['chunk_type']]
    
    # 임베딩 함수 선택
    embedding_functions = {
        'openai': embed_with_openai,
        'sentence-transformer': embed_with_sentence_transformer
    }
    embedding_processor = embedding_functions[config['embedding_module']]

    # LangGraph 노드 함수들
    def process_pdf_node(state: GraphState) -> dict[str, Any]:
        import base64
        
        # base64 디코딩하여 원래 PDF 바이너리 데이터로 변환
        content_base64 = state["messages"][0].content
        content = base64.b64decode(content_base64)
        
        metadata = pdf_processor(content)
        return {"pdf_content": metadata}
    
    def process_chunks_node(state: dict) -> dict[str, Any]:
        metadata = chunk_processor(state["pdf_content"])
        return {"chunks": metadata}
    
    def process_embeddings_node(state: dict) -> dict[str, Any]:
        embeddings, texts, page_info = embedding_processor(state["chunks"])
        return {
            "embeddings": embeddings,
            "texts": texts,
            "page_info": page_info
        }
    
    def store_vectors_node(state: dict) -> dict[str, Any]:
        embedding_type = 'openai' if config['embedding_module'] == 'openai' else 'sentence-transformer'
        file_info = config.get('file_info', {})  # file_info 가져오기
        
        result = store_in_chroma(
            state["embeddings"],
            state["texts"],
            state["page_info"],
            embedding_type=embedding_type,
            file_info=file_info
        )
        return {"vector_store": result}

    # 워크플로우 생성
    workflow = StateGraph(GraphState)
    
    # 노드 추가
    workflow.add_node("process_pdf", process_pdf_node)
    workflow.add_node("process_chunks", process_chunks_node)
    workflow.add_node("process_embeddings", process_embeddings_node)
    workflow.add_node("store_vectors", store_vectors_node)
    
    # 시작 노드 설정
    workflow.set_entry_point("process_pdf")
    
    # 엣지 연결
    workflow.add_edge("process_pdf", "process_chunks")
    workflow.add_edge("process_chunks", "process_embeddings")
    workflow.add_edge("process_embeddings", "store_vectors")
    workflow.add_edge("store_vectors", END)
    
    return workflow.compile()

def load_existing_vectors(collection_name: str) -> Dict:
    """저장된 벡터 컬렉션 로드"""
    import chromadb
    import os
    
    try:
        # Chroma 클라이언트 초기화
        persist_directory = os.path.join(os.getcwd(), "db")
        if not os.path.exists(persist_directory):
            print(f"Database directory not found: {persist_directory}")
            return None
            
        client = chromadb.PersistentClient(path=persist_directory)
        
        # 컬렉션 존재 여부 확인
        collections = client.list_collections()
        if not any(collection.name == collection_name for collection in collections):
            print(f"Collection not found: {collection_name}")
            return None
            
        # 기존 컬렉션 불러오기
        collection = client.get_collection(collection_name)
        
        # 전체 데이터 가져오기
        count = collection.count()
        if count == 0:
            print(f"Collection is empty: {collection_name}")
            return None
            
        results = collection.get()
        
        if not results or "embeddings" not in results:
            print(f"Invalid results format for collection: {collection_name}")
            return None
            
        return {
            "embeddings": results["embeddings"],
            "documents": results["documents"],
            "metadatas": results["metadatas"],
            "ids": results["ids"],
            "collection_name": collection_name,
            "vector_count": count
        }
        
    except Exception as e:
        print(f"Error in load_existing_vectors: {str(e)}")
        return None



def get_available_collections() -> List[Dict]:
    """사용 가능한 벡터 컬렉션 목록 조회"""
    import chromadb
    import os
    from datetime import datetime
    
    persist_directory = os.path.join(os.getcwd(), "db")
    client = chromadb.PersistentClient(path=persist_directory)
    
    collections = []
    try:
        for collection in client.list_collections():
            try:
                collection_data = client.get_collection(collection.name)
                name_parts = collection.name.split('_')
                
                if len(name_parts) >= 7 and name_parts[0] == "vec":
                    collection_info = {
                        "name": collection.name,
                        "vector_count": collection_data.count(),
                        "date": datetime.strptime(name_parts[1], "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S"),
                        "file_name": name_parts[2] + ".pdf",
                        "pdf_module": name_parts[3],
                        "chunk_type": name_parts[4],
                        "chunk_size": name_parts[5],
                        "embedding_type": name_parts[6]  # 변환하지 않고 그대로 표시
                    }
                    collections.append(collection_info)
                
            except Exception as e:
                print(f"Error getting collection info for {collection.name}: {str(e)}")
                continue
                
        return sorted(collections, key=lambda x: x['date'], reverse=True)
    except Exception as e:
        print(f"Error getting collections: {str(e)}")
        return []

def reuse_vectors_for_search(collection_name: str, query: str, n_results: int = 3) -> List[Dict]:
    """저장된 벡터를 사용하여 검색"""
    import chromadb
    import os
    
    # Chroma 클라이언트 초기화
    persist_directory = os.path.join(os.getcwd(), "db")
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        # 컬렉션 가져오기
        collection = client.get_collection(collection_name)
        
        # 임베딩 타입 확인 (컬렉션 이름에서 추출)
        embedding_type = "openai" if "openai" in collection_name else "sentence-transformer"
        
        # 쿼리 임베딩 생성
        if embedding_type == "openai":
            from openai import OpenAI
            openai_client = OpenAI()
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_embedding = response.data[0].embedding
        else:
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
        print(f"Error searching vectors: {str(e)}")
        return []