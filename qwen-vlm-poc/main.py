
import os
import base64
import json
import requests

from pathlib import Path
from dotenv import load_dotenv

# 1. Load API configuration
load_dotenv()

API_KEY = os.getenv("FPT_API_KEY")

BASE_URL = "https://mkp-api.fptcloud.com/v1/chat/completions"
MODEL = "Qwen2.5-VL-7B-Instruct"

IMAGE_PATH = Path("images/crop_test.jpg")
PROMPT_PATH = Path("prompts/traffic_analysis_v3_2.txt")
OUTPUT_PATH = Path("results/response_v3_2.txt")


# 2. Check API key
if not API_KEY:
    raise ValueError("FPT_API_KEY not found in .env")

# 3. Check image
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# 4. Check prompt file
if not PROMPT_PATH.exists():
    raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")

# 5. Convert image to Base64
with open(IMAGE_PATH, "rb") as image_file:
    image_base64 = base64.b64encode(
        image_file.read()
    ).decode("utf-8")

# 5. Create image data URL
image_url = f"data:image/jpeg;base64,{image_base64}"

# 6. Prepare request
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 6. Prepare prompt
with open(
    PROMPT_PATH,
    "r",
    encoding="utf-8"
) as f:
    prompt = f.read()


# 7. Prepare API payload
payload = {
    "model": MODEL,

    "messages": [
        {
            "role": "user",

            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ],

    "temperature": 0.2,

    "max_tokens": 4000,

    "stream": False
}



def validate_vehicles(data: dict) -> list:
    """
    Run deterministic consistency checks on Qwen output.
    Returns a list of human-readable warning strings.
    """
    warnings = []
    vehicles = data.get("vehicles", [])

    if not isinstance(vehicles, list):
        warnings.append("'vehicles' is not a list.")
        return warnings

    valid_positions = {
        "left foreground", "center foreground", "right foreground",
        "left background", "center background", "right background"
    }
    valid_visibility = {"fully_visible", "partially_occluded", "truncated_by_boundary"}

    signature_counts = {}

    for idx, v in enumerate(vehicles):
        vid = v.get("object_id", f"index_{idx}")

        vs = v.get("visibility_status")
        if vs not in valid_visibility:
            warnings.append(f"{vid}: invalid visibility_status '{vs}'.")

        pos = v.get("position", "")
        if pos not in valid_positions:
            warnings.append(f"{vid}: invalid position '{pos}'. Must be one of {sorted(valid_positions)}.")

        edges = v.get("truncated_edges", [])
        if not isinstance(edges, list):
            warnings.append(f"{vid}: truncated_edges is not a list.")
            edges = []

        if "left" in edges and not pos.startswith("left"):
            warnings.append(f"{vid}: truncated_edges contains 'left' but position is '{pos}'.")
        if "right" in edges and not pos.startswith("right"):
            warnings.append(f"{vid}: truncated_edges contains 'right' but position is '{pos}'.")
        if pos.startswith("center") and ("left" in edges or "right" in edges):
            warnings.append(f"{vid}: position is center but truncated_edges is {edges}.")

        # Hard boundary rules: vehicles at image edges are likely truncated
        if pos.startswith("right") and vs == "fully_visible":
            warnings.append(f"{vid}: position is right edge but visibility_status is fully_visible; "
                            f"right-edge vehicles are frequently truncated by image border.")
        if pos.startswith("left") and vs == "fully_visible":
            warnings.append(f"{vid}: position is left edge but visibility_status is fully_visible; "
                            f"left-edge vehicles may be truncated by image border.")
        if pos.startswith("right") and "right" not in edges:
            warnings.append(f"{vid}: position is right edge but truncated_edges does not contain 'right'; "
                            f"verify whether the vehicle is cut by the right border.")
        if pos.startswith("left") and "left" not in edges:
            warnings.append(f"{vid}: position is left edge but truncated_edges does not contain 'left'; "
                            f"verify whether the vehicle is cut by the left border.")
        if v.get("class") == "car" and pos == "right foreground" and not edges:
            warnings.append(f"{vid}: car at right foreground with no truncated_edges is highly suspicious; "
                            f"right-edge cars in traffic images are commonly truncated.")

        evidence = v.get("evidence", "") or ""
        occluder = v.get("occluder")
        occlusion_keywords = ["hidden by", "occluded by", "blocked by"]
        if any(kw in evidence.lower() for kw in occlusion_keywords) and occluder is None:
            warnings.append(f"{vid}: evidence mentions occlusion but occluder is null.")

        sig = (
            v.get("class"),
            tuple(v.get("visible_parts", [])),
            tuple(v.get("hidden_parts", [])),
            tuple(v.get("truncated_edges", [])),
        )
        signature_counts[sig] = signature_counts.get(sig, 0) + 1

    for sig, count in signature_counts.items():
        if count > 1:
            cls, parts_visible, parts_hidden, edges = sig
            warnings.append(
                f"Repetition: {count} vehicles share identical "
                f"class='{cls}', visible_parts={list(parts_visible)}, "
                f"hidden_parts={list(parts_hidden)}, "
                f"truncated_edges={list(edges)}. "
                f"This may indicate copy-paste instead of per-vehicle analysis."
            )

    return warnings

# 7. Send request
print("Sending image to Qwen2.5-VL...")

try:
    response = requests.post(
        BASE_URL,
        headers=headers,
        json=payload,
        timeout=180
    )

    response.raise_for_status()
    result = response.json()

    # Support both direct OpenAI-style and wrapped responses
    data = result.get("data", result)
    choices = data.get("choices", [])

    if not choices:
        print("Unexpected API response:")
        print(result)
        raise RuntimeError("No model output found.")

    answer = choices[0]["message"]["content"]

    # 8. Strip markdown code block fences if present (e.g. ```json ... ```)
    cleaned_answer = answer.strip()
    if cleaned_answer.startswith("```"):
        cleaned_answer = cleaned_answer.split("\n", 1)[1] if "\n" in cleaned_answer else cleaned_answer
        if cleaned_answer.endswith("```"):
            cleaned_answer = cleaned_answer.rsplit("```", 1)[0]
    cleaned_answer = cleaned_answer.strip()

    # 9. Validate JSON before saving
    try:
        parsed_json = json.loads(cleaned_answer)
    except json.JSONDecodeError as e:
        print("ERROR: Model output is not valid JSON.")
        print("Raw answer:\n", answer)
        print("Cleaned answer:\n", cleaned_answer)
        raise

    # 10. Post-processing validation
    validation_warnings = validate_vehicles(parsed_json)

    if validation_warnings:
        print("\n========== VALIDATION WARNINGS ==========\n")
        for w in validation_warnings:
            print(f"- {w}")
        print("\n")

    # Attach validation report to output for traceability
    parsed_json["_validation"] = {
        "warnings": validation_warnings,
        "prompt_file": str(PROMPT_PATH),
        "image_file": str(IMAGE_PATH)
    }

    # 11. Display result
    print("\n========== QWEN RESPONSE ==========\n")
    print(json.dumps(parsed_json, ensure_ascii=False, indent=2))

    # 12. Save result
    os.makedirs("results", exist_ok=True)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(parsed_json, f, ensure_ascii=False, indent=2)

    print(f"\nResult saved to {OUTPUT_PATH}")

except requests.exceptions.HTTPError:
    print("HTTP Error:", response.status_code)
    print(response.text)

except requests.exceptions.RequestException as e:
    print("Connection error:", e)

except Exception as e:
    print("Error:", e)
