# import os
# import argparse
# from dotenv import load_dotenv
# from pinecone import Pinecone
# from openai import OpenAI

# load_dotenv()

# # CLI flags
# parser = argparse.ArgumentParser(description="Query Pinecone and inspect results")
# parser.add_argument("--show-values", action="store_true", help="Print the first N components of the vector values for each match")
# parser.add_argument("--values-head", type=int, default=8, help="Number of vector components to print when --show-values is set")
# parser.add_argument("--content-chars", type=int, default=1240, help="Number of characters of content to print per match")
# args = parser.parse_args()

# # Use the SAME normalized name that the ingester logs (it normalizes env name to lowercase + hyphen only)
# index_name = os.getenv("PINECONE_INDEX_NAME", "lipstick-chatbot").lower().replace("_", "-")
# namespace = os.getenv("PINECONE_NAMESPACE", "default")

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index = pc.Index(index_name)

# # Build a test query embedding using same OpenAI model
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# # Example queries you can try (attributes and Q&A content)
# queries = [
#     "HEX Code and RGB values in the color attributes",
#     "classification fields for lipstick taxonomy",
#     "Launch Type options in the schema",
#     "Glossier Generation G Jam finish and opacity",
# ]

# for q in queries:
#     emb = client.embeddings.create(model=model, input=q).data[0].embedding
#     res = index.query(
#         namespace=namespace,
#         vector=emb,
#         top_k=5,
#         include_values=True,
#         include_metadata=True,
#     )
#     print("\n--- Query:", q)
#     for i, m in enumerate(res["matches"], start=1):
#         md = m.get("metadata", {})
#         values = m.get("values", [])
#         print(f"{i}. id={m['id']} score={m['score']:.4f} dim={len(values)}")
#         if args.show_values:
#             head_n = max(0, args.values_head)
#             print(f"   values_head[{head_n}]:", values[:head_n])
#         # SHOW the embedded text body (we upserted this into metadata.content)
#         content = md.get("content") or ""
#         print("   content:", content[:args.content_chars].replace("\n", " "))
#         # If WHY is present, print a dedicated line to make it visible regardless of truncation
#         if "WHY:" in content:
#             why_part = content.split("WHY:", 1)[1]
#             # stop at SOLUTION if present to keep WHY focused
#             if "SOLUTION:" in why_part:
#                 why_part = why_part.split("SOLUTION:", 1)[0]
#             print("   WHY:", why_part.strip().replace("\n", " ")[:args.content_chars])
#         # Optional: show slimmed metadata keys you care about
#         for k in ("product_id", "product_name", "brand", "group", "doc_type", "topic"):
#             if k in md:
#                 print(f"   {k}:", md[k])

#!/usr/bin/env python3
# """
# Check whether 'Ultimate_Lipstick_Attribute_System_Indian.json' is stored in Pinecone.

# What it does:
# - describe_index_stats(filter=...) to count vectors for this file (by namespace)
# - a filtered semantic query to print a few sample records (ids + metadata)

# Env:
#   PINECONE_API_KEY=...
#   OPENAI_API_KEY=...
#   PINECONE_INDEX_NAME=products-general
#   PINECONE_NAMESPACE=default
#   OPENAI_EMBEDDING_MODEL=text-embedding-3-large
# """

# import os
# from dotenv import load_dotenv
# from typing import Optional, Dict, Any

# from pinecone import Pinecone
# from pinecone.exceptions.exceptions import NotFoundException, PineconeApiException
# from openai import OpenAI

# """
# Load environment variables from a .env file (if present) so that
# PINECONE_API_KEY, OPENAI_API_KEY, and other settings are available.
# """
# load_dotenv()

# FILENAME = "Ultimate_Lipstick_Attribute_System_Indian.json"

# def get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
#     v = os.getenv(name, default)
#     if required and v is None:
#         raise RuntimeError(f"Missing env var: {name}")
#     return v

# def vector_count_from_stats(stats: Dict[str, Any], namespace: str) -> int:
#     # Pinecone returns stats in stats["namespaces"][ns]["vector_count"] (or "vectorCount")
#     ns = (stats.get("namespaces") or {}).get(namespace, {})
#     return ns.get("vector_count") or ns.get("vectorCount") or 0

# def main():
#     pc = Pinecone(api_key=get_env("PINECONE_API_KEY", required=True))
#     # Normalize index name the same way the ingester does
#     raw_index_name = get_env("PINECONE_INDEX_NAME", "Lipstick-chatbot")
#     index_name = raw_index_name.lower().replace("_", "-")
#     namespace = get_env("PINECONE_NAMESPACE", "default")

#     print(f"[info] Using Pinecone index='{index_name}' (from '{raw_index_name}') namespace='{namespace}'")
#     try:
#         index = pc.Index(index_name)
#     except NotFoundException:
#         raise SystemExit(
#             f"[error] Pinecone index '{index_name}' not found.\n"
#             f"- Ensure the index exists in your Pinecone project and region.\n"
#             f"- If your .env has 'PINECONE_INDEX_NAME={raw_index_name}', note it is normalized to '{index_name}'.\n"
#             f"- List your indexes in the Pinecone Console or adjust PINECONE_INDEX_NAME in .env accordingly."
#         )

#     # --- 1) Count vectors for this file via index stats (no embedding needed)
#     # Note: Serverless/Starter indexes do NOT support metadata filtering in describe_index_stats.
#     # We try; if unsupported, we continue without count and just run a filtered query.
#     count = None
#     try:
#         stats = index.describe_index_stats(
#             filter={"source": {"$eq": FILENAME}}
#         )
#         count = vector_count_from_stats(stats, namespace)
#         print(f"[stats] namespace='{namespace}'  file='{FILENAME}'  vector_count={count}")
#     except PineconeApiException as e:
#         # Serverless limitation: code=3, message about not supporting metadata filtering
#         msg = getattr(e, 'body', None) or str(e)
#         print(f"[warn] describe_index_stats with metadata filter not supported on this index: {msg}")
#         print("[warn] Skipping count; proceeding to run a filtered semantic query to verify presence.")

#     # --- 2) Query a few examples using progressively broader filters to verify presence
#     if count is None or count > 0:
#         oai = OpenAI(api_key=get_env("OPENAI_API_KEY", required=True))
#         embed_model = get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
#         # Any generic query text works; we use a schema-oriented prompt
#         qtext = "lipstick attribute schema: performance metrics, application techniques, accessibility"
#         qvec = oai.embeddings.create(model=embed_model, input=[qtext]).data[0].embedding

#         attempts = [
#             ("source==FILENAME", {"source": {"$eq": FILENAME}}),
#             ("doc_type=='attributes'", {"doc_type": {"$eq": "attributes"}}),
#             ("group in key attribute groups",
#              {"group": {"$in": [
#                  "classification",
#                  "color_attributes",
#                  "indian_skin_tone_mapping",
#                  "performance_metrics",
#                  "application_techniques",
#                  "accessibility",
#              ]}}),
#         ]

#         found_any = False
#         for label, flt in attempts:
#             try:
#                 res = index.query(
#                     namespace=namespace,
#                     vector=qvec,
#                     top_k=5,
#                     include_metadata=True,
#                     filter=flt,
#                 )
#             except Exception as e:
#                 print(f"[warn] Query with filter '{label}' failed: {e}")
#                 continue
#             matches = res.get("matches") or getattr(res, "matches", []) or []
#             print(f"[sample:{label}] showing {len(matches)} match(es):\n")
#             for m in matches:
#                 md = m.get("metadata", {})
#                 print(f"- id: {m.get('id')}")
#                 print(f"  score: {m.get('score')}")
#                 print(f"  doc_type: {md.get('doc_type')}")
#                 print(f"  group/section: {md.get('group') or md.get('section')}")
#                 print(f"  product_id: {md.get('product_id')}")
#                 print(f"  product_name: {md.get('product_name')}")
#                 print(f"  brand: {md.get('brand')}")
#                 if "source" in md:
#                     print(f"  source: {md.get('source')}")
#                 content = (md.get('content') or "")[:180].replace("\n", " ")
#                 print(f"  content_snippet: {content}...")
#                 print()
#             if matches:
#                 found_any = True
#         if not found_any:
#             print("[info] No matches found with verification filters. If you need file-level certainty, re-ingest with INCLUDE_DEBUG_METADATA=1 to store metadata.source and retry.")
#     else:
#         print("[info] No vectors found for that file in this namespace.")

# if __name__ == "__main__":
#     main()


# ==================== DUPES DATA VERIFICATION ====================

def verify_dupes_data():
    """
    Verify that dupes data from 'nars-dolce-vita-dupes.json' is properly stored in Pinecone.
    
    What it does:
    - Connects to Pinecone index using environment variables
    - Runs semantic queries to find dupe records
    - Shows sample records with metadata for verification
    
    Env variables needed:
      PINECONE_API_KEY=...
      OPENAI_API_KEY=...
      PINECONE_INDEX_NAME=lipstick-chatbot
      PINECONE_NAMESPACE=default (or dupes)
      OPENAI_EMBEDDING_MODEL=text-embedding-3-large
    """
    import os
    from dotenv import load_dotenv
    from pinecone import Pinecone
    from openai import OpenAI
    
    load_dotenv()
    
    # Configuration
    index_name = os.getenv("PINECONE_INDEX_NAME", "lipstick-chatbot").lower().replace("_", "-")
    namespace = os.getenv("PINECONE_NAMESPACE", "default")
    
    print(f"[DUPES VERIFICATION] Using index='{index_name}' namespace='{namespace}'")
    
    # Connect to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    # Connect to OpenAI
    oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    
    # Test queries specifically for dupes data
    test_queries = [
        "NARS Air Matte Lip Color Dolce Vita dupes alternatives",
        "MAC Mehr lipstick dupe comparison",
        "drugstore lipstick dupes affordable alternatives", 
        "best budget lipstick dupes under $10",
        "matte lipstick formula comparison pros cons"
    ]
    
    print("\n" + "="*60)
    print("TESTING DUPES DATA RETRIEVAL")
    print("="*60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: {query}")
        
        # Generate embedding for query
        qvec = oai.embeddings.create(model=embed_model, input=[query]).data[0].embedding
        
        # Search with filter for dupes data
        try:
            res = index.query(
                namespace=namespace,
                vector=qvec,
                top_k=5,
                include_metadata=True,
                filter={"doc_family": {"$eq": "dupe_guide"}}
            )
            
            matches = res.get("matches", [])
            print(f"Found {len(matches)} dupe records:")
            
            for j, match in enumerate(matches, 1):
                md = match.get("metadata", {})
                score = match.get("score", 0)
                
                print(f"\n  {j}. Score: {score:.4f}")
                print(f"     ID: {match.get('id', 'N/A')}")
                print(f"     Product: {md.get('product_name', 'N/A')}")
                print(f"     Brand: {md.get('brand', 'N/A')}")
                print(f"     Section: {md.get('section', 'N/A')}")
                
                # Show dupe-specific fields if available
                if md.get('dupe_brand'):
                    print(f"     Dupe Brand: {md.get('dupe_brand')}")
                    print(f"     Dupe Product: {md.get('dupe_product', 'N/A')}")
                    print(f"     Dupe Shade: {md.get('dupe_shade', 'N/A')}")
                    print(f"     Rank: {md.get('rank', 'N/A')}")
                    print(f"     Price: ${md.get('price_usd_dupe', 'N/A')}")
                
                # Show content snippet
                content = md.get('content', '')
                if content:
                    snippet = content[:200].replace('\n', ' ')
                    print(f"     Content: {snippet}...")
                    
        except Exception as e:
            print(f"     ERROR: {e}")
    
    # Additional verification: Check for specific product ID
    print(f"\n" + "="*60)
    print("VERIFYING SPECIFIC PRODUCT DATA")
    print("="*60)
    
    try:
        # Query for the specific NARS product
        qvec = oai.embeddings.create(
            model=embed_model, 
            input=["NARS Air Matte Lip Color Dolce Vita"]
        ).data[0].embedding
        
        res = index.query(
            namespace=namespace,
            vector=qvec,
            top_k=10,
            include_metadata=True,
            filter={"product_id": {"$eq": "nars_air_matte_lip_color_-_dolce_vita"}}
        )
        
        matches = res.get("matches", [])
        print(f"\nFound {len(matches)} records for product_id='nars_air_matte_lip_color_-_dolce_vita'")
        
        # Group by section type
        sections = {}
        for match in matches:
            md = match.get("metadata", {})
            section = md.get('section', 'unknown')
            if section not in sections:
                sections[section] = []
            sections[section].append(match)
        
        for section, section_matches in sections.items():
            print(f"\n  Section '{section}': {len(section_matches)} records")
            for match in section_matches[:3]:  # Show first 3 of each section
                md = match.get("metadata", {})
                print(f"    - {match.get('id', 'N/A')[:50]}...")
                if md.get('dupe_brand'):
                    print(f"      Dupe: {md.get('dupe_brand')} {md.get('dupe_product')}")
                
    except Exception as e:
        print(f"ERROR in product verification: {e}")
    
    print(f"\n" + "="*60)
    print("DUPES VERIFICATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Uncomment the function you want to run:
    # main()  # Original verification
    verify_dupes_data()  # New dupes verification
