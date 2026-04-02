import os
from dotenv import load_dotenv
import asyncio
import argparse
import fitz  # PyMuPDF
from llama_parse import LlamaParse
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
import glob

# --- STEP 1: EXTRACT TEXT (LlamaParse) ---
async def parse_document_with_llamaparse(file_path, api_key):
    print(f"🔄 Parsing {file_path} with LlamaParse...")
    os.environ["LLAMA_CLOUD_API_KEY"] = api_key
    
    parser = LlamaParse(result_type="markdown", verbose=True, language="en")
    try:
        documents = await parser.aload_data(file_path)
        print(f"✅ Parsing complete.")
        return documents
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        return []

# --- STEP 2: EXTRACT & CAPTION DIAGRAMS (Hugging Face) ---

async def process_diagrams(pdf_path, documents, processor, model, device="cpu", output_dir=None):
    """Diagram captioning is disabled; return no captions."""
    print("ℹ️  Diagram captioning is disabled; skipping image caption generation.")
    return {}

# --- STEP 3: MERGE & INDEX (FIXED) ---
# --- STEP 3: MERGE & INDEX (Simplified & Robust) ---
def merge_captions_into_documents(documents, image_captions):
    print("🔗 Merging diagram captions into text context...")
    enriched_docs = []
    
    for doc in documents:
        page_num = int(doc.metadata.get('page_label', 1)) - 1
        caption_text = image_captions.get(page_num, "")
        
        if caption_text:
            # ✅ Create NEW Document with ONLY text + metadata (safe across versions)
            new_text = doc.text + "\n\n---\n" + caption_text
            enriched_doc = Document(
                text=new_text,
                metadata=doc.metadata.copy() if doc.metadata else {}
            )
            enriched_docs.append(enriched_doc)
        else:
            # No captions to add, keep original doc
            enriched_docs.append(doc)
    
    return enriched_docs

async def build_index(documents, embed_model_name="BAAI/bge-small-en-v1.5"):
    print("📚 Building Vector Index...")
    
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    
    node_parser = SemanticSplitterNodeParser(
        buffer_size=1, 
        breakpoint_percentile_threshold=95,
        embed_model=embed_model
    )
    
    nodes = node_parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    print("✅ Index built successfully.")
    return index

# --- UTILITY: SAVE DOCUMENTS TO FILE ---
def save_documents_to_file(documents, output_path, format="txt"):
    """Save enriched documents to a file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    if format == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            for i, doc in enumerate(documents):
                page_num = doc.metadata.get('page_label', i+1)
                f.write(f"\n{'='*60}\n")
                f.write(f"📄 PAGE {page_num}\n")
                f.write(f"{'='*60}\n\n")
                f.write(doc.text)
                f.write("\n\n")
        print(f"✅ Saved {len(documents)} documents to {output_path}")
        
    elif format == "json":
        # Structured format: better for re-loading or debugging
        data = [
            {
                "page": doc.metadata.get('page_label', i+1),
                "text": doc.text,
                "metadata": doc.metadata
            }
            for i, doc in enumerate(documents)
        ]
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(documents)} documents to {output_path} (JSON)")
        
    else:
        raise ValueError(f"Unsupported format: {format}")

# --- MAIN EXECUTION ---
async def process_single_pdf(input_file, output_dir, processor, model, llama_api_key, embed_model_name):
    """Process a single PDF file and save outputs with derived filenames."""
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found.")
        return
    
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    os.makedirs(output_dir, exist_ok=True)

    documents = await parse_document_with_llamaparse(input_file, llama_api_key)
    if not documents: 
        print(f"⚠️ No documents parsed from {input_file}")
        return

    enriched_docs = merge_captions_into_documents(documents, {})

    # Output files with derived names
    output_txt = os.path.join(output_dir, f"{base_filename}_enriched.txt")
    output_json = os.path.join(output_dir, f"{base_filename}_enriched.json")
    
    save_documents_to_file(enriched_docs, output_txt, format="txt")
    save_documents_to_file(enriched_docs, output_json, format="json")
    print(f"✅ Completed processing: {input_file}")

async def process_multiple_pdfs(input_path, output_dir, processor, model, llama_api_key, embed_model_name):
    """Process all PDFs in a folder."""
    if os.path.isfile(input_path):
        # Single file
        await process_single_pdf(input_path, output_dir, processor, model, llama_api_key, embed_model_name)
    else:
        # Directory - find all PDFs
        pdf_files = glob.glob(os.path.join(input_path, "*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {input_path}")
            return
        
        print(f"🔍 Found {len(pdf_files)} PDF file(s) to process...")
        for pdf_file in pdf_files:
            print(f"\n📄 Processing: {os.path.basename(pdf_file)}")
            await process_single_pdf(pdf_file, output_dir, processor, model, llama_api_key, embed_model_name)

async def main(input_path, output_dir, llama_api_key, embed_model_name, hf_model_name="Salesforce/blip-image-captioning-base"):
    # Determine if input is file or folder
    if not os.path.exists(input_path):
        print(f"❌ Error: Path '{input_path}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Process single or multiple PDFs
    await process_multiple_pdfs(input_path, output_dir, None, None, llama_api_key, embed_model_name)

if __name__ == "__main__":
    load_dotenv() 

    parser = argparse.ArgumentParser(description="PDF to LLM Index Pipeline with Hugging Face Image Captioning (Single or Batch)")
    
    parser.add_argument("-i", "--input", required=True, help="Path to a PDF file or folder containing PDF files")
    parser.add_argument("-o", "--output", default="processed_data", help="Output directory for enriched documents")
    parser.add_argument("--hf-model", default="Salesforce/blip-image-captioning-base", help="Hugging Face model ID for image captioning")
    parser.add_argument("--llama-key", default=os.getenv("LLAMA_CLOUD_API_KEY"), help="LlamaCloud API Key")
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5", 
                       help="Embedding model name (HuggingFace)")
    
    args = parser.parse_args()
    
    asyncio.run(main(
        input_path=args.input,
        output_dir=args.output,
        llama_api_key=args.llama_key,
        embed_model_name=args.embed_model,
        hf_model_name=args.hf_model
    ))