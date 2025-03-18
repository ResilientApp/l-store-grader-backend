from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from english_words import get_english_words_set
from random import sample
from datetime import datetime
from io import BytesIO
import uuid
import subprocess
import json
import zipfile
import os
import random
from urllib.parse import quote_plus
from pymongo import MongoClient
from dotenv import load_dotenv
import hashlib
import asyncio
import aiohttp
import time
import requests

# 1) ADD gql imports
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

# 2) IMPORT nest_asyncio AND APPLY
import nest_asyncio
nest_asyncio.apply()

# ----------------- Milestone Config -----------------
"""
This dictionary controls which milestones are enabled and whether the extended version
is enabled for each milestone. Adjust these boolean values to enable/disable as desired.
"""
MILESTONE_CONFIG = {
    "milestone1": {"enabled": True, "extended_enabled": True},
    "milestone2": {"enabled": True, "extended_enabled": True},
    "milestone3": {"enabled": True, "extended_enabled": False},
}

# Load environment variables from .env file
load_dotenv()

# Retrieve variables
username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
deepseek_url = os.getenv('DEEPSEEK_URL')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
resdb_url = os.getenv('RESILIENTDB_URL')
resdb_private_key = os.getenv('RESILIENTDB_PRIVATE_KEY')
resdb_public_key = os.getenv('RESILIENTDB_PUBLIC_KEY')
resdb_recipient_key = os.getenv('RESILIENTDB_RECIPIENT_KEY')

encoded_username = quote_plus(username)
encoded_password = quote_plus(password)

# Construct MongoDB URI
uri = f"mongodb+srv://{encoded_username}:{encoded_password}@{host}/{db_name}?retryWrites=true&w=majority"
client = MongoClient(uri)
db = client[db_name]

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.expanduser('~/l-store-grader')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def generate_unique_name():
    """
    Generates a random 3-word name: e.g. "Quiet Heart Lane"
    """
    from english_words import get_english_words_set
    words_list = list(get_english_words_set(['web2'], lower=True, alpha=True))
    filtered_words_list = [w for w in words_list if len(w) <= 5]
    chosen = random.sample(filtered_words_list, 3)
    chosen = [word.capitalize() for word in chosen]
    return " ".join(chosen)


# ----------------- AI & Commit Extraction functions ------------------
async def read_python_files(folder_path):
    code_data = ""

    async def read_file(file_path, filename):
        try:
            content = await asyncio.to_thread(
                lambda: open(file_path, 'r', encoding='utf-8', errors='replace').read()
            )
            return filename, content
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return filename, None

    tasks = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                tasks.append(read_file(file_path, file))

    results = await asyncio.gather(*tasks)
    for filename, content in results:
        if content:
            code_data += f"\n# File: {filename}\n" + content + "\n"
    return code_data


async def call_deepseek_api(prompt, model="deepseek-reasoner",
                            api_key=f"{deepseek_api_key}",
                            total_timeout=300):
    """
    Calls the DeepSeek API with a configurable timeout (default=300 seconds).
    """
    url = f"{deepseek_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a code analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    timeout = aiohttp.ClientTimeout(total=total_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            start_time = time.time()
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
            end_time = time.time()
            result["request_time"] = round(end_time - start_time, 4)
            return result
        except aiohttp.ClientError as e:
            return {"error": f"API call failed: {e}"}


def parse_response_content(response):
    try:
        content = response["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            lines = content.splitlines()
            lines = lines[1:] if len(lines) > 1 else lines
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)

        if not content.startswith("{"):
            content = "{" + content
        if not content.endswith("}"):
            content = content + "}"
        return json.loads(content)
    except Exception as e:
        print("Error parsing AI response:", e)
        return {}


async def run_ai_tests(lstore_folder, ai_timeout=300):
    """
    Runs multiple AI checks with a user-specified or default 300-second timeout.
    """
    code_data = await read_python_files(lstore_folder)

    prompt_ai = f"""
    Analyze the code below and determine the likelihood it was generated by AI. Return a confidence score between 0 and 1.
    Output only:
    ```json
    "ai_generated_confidence": score
    ```
    Here is the code:
    ```python
    {code_data}
    ```
    """

    prompt_pickle = f"""
    Analyze the code below and check if the **pickle** module is used to serialize "tail" and "base" pages to disk. Return a higher confidence score if so.
    Output only a JSON object with:
    ```json
    "pickle_serialization_confidence": score
    ```
    Here is the code:
    ```python
    {code_data}
    ```
    """

    prompt_struct = f"""
    Analyze the code below and check if the **struct** module is used to serialize "tail" and "base" pages to disk. Return a higher confidence score if so.
    Output only a JSON object with:
    ```json
    "struct_serialization_confidence": score
    ```
    Here is the code:
    ```python
    {code_data}
    ```
    """

    prompt_json_module = f"""
    Analyze the code below and check if the **json** module is used to serialize "tail" and "base" pages to disk. Return a higher confidence score if so.
    Output only a JSON object with:
    ```json
    "json_serialization_confidence": score
    ```
    Here is the code:
    ```python
    {code_data}
    ```
    """

    overall_start = time.time()
    tasks = [
        call_deepseek_api(prompt_ai, total_timeout=ai_timeout),
        call_deepseek_api(prompt_pickle, total_timeout=ai_timeout),
        call_deepseek_api(prompt_struct, total_timeout=ai_timeout),
        call_deepseek_api(prompt_json_module, total_timeout=ai_timeout),
    ]
    responses = await asyncio.gather(*tasks)

    parsed_ai = parse_response_content(responses[0])
    parsed_pickle = parse_response_content(responses[1])
    parsed_struct = parse_response_content(responses[2])
    parsed_json = parse_response_content(responses[3])

    overall_end = time.time()
    return {
        "ai_generated_confidence": parsed_ai.get("ai_generated_confidence"),
        "pickle_serialization_confidence": parsed_pickle.get("pickle_serialization_confidence"),
        "struct_serialization_confidence": parsed_struct.get("struct_serialization_confidence"),
        "json_serialization_confidence": parsed_json.get("json_serialization_confidence"),
        "individual_request_times": {
            "ai_generated": responses[0].get("request_time"),
            "pickle": responses[1].get("request_time"),
            "struct": responses[2].get("request_time"),
            "json": responses[3].get("request_time"),
        },
        "total_time": round(overall_end - overall_start, 2)
    }


def extract_commit_stats(folder_path):
    stats = {
        "local_git_commits": [],
        "contributors": []  # We'll populate
    }

    git_path = os.path.join(folder_path, ".git")
    if os.path.exists(git_path):
        try:
            output = subprocess.check_output(
                ["git", "--no-pager", "log", "--pretty=format:%h - %an, %ad", "--date=short"],
                cwd=folder_path
            )
            lines = output.decode("utf-8").splitlines()
            stats["local_git_commits"] = lines

            # Count commits per author
            author_counts = {}
            for line in lines:
                parts = line.split('-', 1)
                if len(parts) < 2:
                    continue
                after_dash = parts[1].strip()   # e.g. "Apratim Shukla, 2025-03-11"
                comma_idx = after_dash.rfind(',')
                if comma_idx != -1:
                    author_name = after_dash[:comma_idx].strip()
                else:
                    author_name = after_dash
                author_counts[author_name] = author_counts.get(author_name, 0) + 1

            stats["contributors"] = [
                {"name": name, "commits": count}
                for name, count in author_counts.items()
            ]

        except Exception as e:
            stats["error"] = f"Error extracting git commit stats: {e}"

    return stats


def merge_contributors(a, b):
    """ Merge two arrays of {name, commits}, summing commits for duplicates. """
    merged_map = {}
    for obj in a:
        merged_map[obj["name"]] = merged_map.get(obj["name"], 0) + obj["commits"]
    for obj in b:
        merged_map[obj["name"]] = merged_map.get(obj["name"], 0) + obj["commits"]
    return [{"name": k, "commits": v} for k, v in merged_map.items()]


def milestone_tests(extract_path, lstore_path, milestone_name, timeout_val=60):
    """
    Runs the single official milestone test script with the given 'timeout_val' (in seconds).
    """
    milestones_collection = db['new_milestones']
    milestone_script = milestones_collection.find_one({"milestone": milestone_name})

    default_metrics = {
        "insert_time": 0,
        "update_time": 0,
        "select_time": 0,
        "agg_time": 0,
        "delete_time": 0
    }

    def default_failure(message):
        tests = {"Test Execution": f"Failed: {message}"}
        return default_metrics, tests, 0, 1

    if not milestone_script:
        print("Milestone script not found for milestone:", milestone_name)
        return default_failure("Milestone script not found.")

    script_code = milestone_script['code']
    tester_script_path = os.path.join(extract_path, 'tester.py')

    try:
        with open(tester_script_path, 'w') as f:
            f.write(script_code)
    except Exception as e:
        print("Error writing tester.py:", e)
        return default_failure("Could not write tester.py file.")

    env = os.environ.copy()
    env['PYTHONPATH'] = lstore_path + os.pathsep + env.get('PYTHONPATH', '')

    try:
        result = subprocess.run(
            ["python", tester_script_path],
            capture_output=True,
            text=True,
            env=env,
            cwd=extract_path,
            timeout=timeout_val
        )
    except subprocess.TimeoutExpired:
        print("Tester script timed out.")
        return default_failure("Test execution timed out.")
    except Exception as e:
        print("Subprocess error:", e)
        return default_failure(f"Subprocess error: {e}")

    if result.returncode != 0:
        print("Tester script stderr:", result.stderr)
        error_message = result.stderr.strip() or "Test execution failed with non-zero return code."
        return default_failure(error_message)

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Invalid JSON tester output:", result.stdout)
        return default_failure("Invalid JSON output from tester script.")

    tests_out = output.get("tests", {})
    if "count" not in output or "total" not in output:
        total = len(tests_out)
        count = sum(1 for outcome in tests_out.values() if not outcome.startswith("Error"))
    else:
        count = output["count"]
        total = output["total"]

    results_out = output.get("results", default_metrics)
    results_out = {
        k: round(v, 3) if isinstance(v, (int, float)) else v
        for k, v in results_out.items()
    }

    return results_out, tests_out, count, total


POST_TRANSACTION_MUTATION = gql(f"""
mutation postTxn($asset: JSONScalar!) {{
  postTransaction(
    data: {{
      operation: "CREATE",
      amount: 100,
      signerPublicKey: "{resdb_public_key}",
      signerPrivateKey: "{resdb_private_key}",
      recipientPublicKey: "{resdb_recipient_key}",
      asset: $asset
    }}
  ) {{
    id
  }}
}}
""")

async def _store_on_resilientdb_async(result_json):
    transport = AIOHTTPTransport(url=f"{resdb_url}/graphql")
    client = Client(transport=transport, fetch_schema_from_transport=False)

    asset_payload = {"data": result_json}

    try:
        response = await client.execute_async(
            POST_TRANSACTION_MUTATION,
            variable_values={"asset": asset_payload},
        )
        return response.get("postTransaction", {}).get("id")
    except Exception as e:
        print(f"GraphQL Mutation Error: {e}")
        return None


def store_on_resilientdb(result_json):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(_store_on_resilientdb_async(result_json))
    except Exception as e:
        print("Error storing on resilientdb:", e)
        return None


@app.route('/results/<transaction_id>', methods=['GET'])
def get_results_transaction(transaction_id):
    query_str = f'''
    query {{
      getTransaction(id: "{transaction_id}") {{
        id
        asset
      }}
    }}
    '''
    try:
        response = requests.post(f"{resdb_url}/graphql", json={"query": query_str})
        if response.status_code == 200:
            data = response.json()
            asset_data = data.get("data", {}).get("getTransaction", {}).get("asset", {})
            return jsonify({
                "resilientdb_tx_id": transaction_id,
                "resilientdb_result": asset_data
            })
        else:
            return jsonify({"error": "Error fetching transaction data from resilientdb."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    """
    Displays each milestone-specific submission, sorted by:
      - Highest 'passed' first,
      - Then minimal 'total_time' ascending.

    Usage: GET /leaderboard?milestone=<milestoneName>
    """
    milestone = request.args.get("milestone")
    if not milestone:
        return jsonify({"error": "Missing milestone param"}), 400

    # Query only docs matching that milestone:
    docs = list(db["new_lstore"].find({"milestone": milestone}))

    leaderboard_data = []
    for d in docs:
        submission_name = d.get("submission_name", "Unnamed Submission")
        passed = d.get("passed", 0)
        total_tests = d.get("total_tests", 0)
        tx_id = d.get("resilientdb_tx_id", None)  # so front-end can link to it

        perf = d.get("performance_results", {})
        # Summation of times
        perf_sum = 0
        for key, val in perf.items():
            if key.endswith("_time") and isinstance(val, (int, float)):
                perf_sum += val

        # Build the entry
        entry = {
            "name": submission_name,
            "count": passed,        # test cases passed
            "total": total_tests,
            "total_time": round(perf_sum, 3),
            "tx_id": tx_id         # needed for /results/<tx_id>
        }
        leaderboard_data.append(entry)

    # Sort: 1) passed descending, 2) total_time ascending
    leaderboard_data.sort(key=lambda x: (-x["count"], x["total_time"]))

    return jsonify(leaderboard_data)


@app.route('/results', methods=['GET', 'POST'])
def show_results():
    """
    Main route to run tests and store the results.
    """
    # Start timing now (for total sub_process time).
    overall_start = time.time()

    if request.method == 'POST':
        milestone = request.form.get('milestone')
        timeout_param = request.form.get('timeout')
        file = request.files.get('file')
        github_repo = request.form.get('github_repo')
        ai_param = request.form.get('ai', 'false').lower() == 'true'
        extended_param = request.form.get('extended', 'false')
        submission_name = request.form.get('submission_name')
    else:
        milestone = request.args.get('milestone')
        timeout_param = request.args.get('timeout')
        file = None
        github_repo = None
        ai_param = False
        extended_param = request.args.get('extended', 'false')
        submission_name = None

    if not submission_name:
        submission_name = generate_unique_name()

    # -- Convert user-provided test timeout (in minutes) to seconds
    try:
        if timeout_param:
            timeout_val = float(timeout_param) * 60
        else:
            timeout_val = 60
    except:
        timeout_val = 60

    # -- Convert user-provided AI test timeout (in seconds)
    #    If not provided, default to 300 seconds
    try:
        if timeout_param:
            ai_timeout_val = float(timeout_param) * 60
        else:
            ai_timeout_val = 300
    except:
        ai_timeout_val = 300

    if not milestone:
        return jsonify({"error": "No milestone provided."}), 400

    # Validate milestone against MILESTONE_CONFIG
    if milestone not in MILESTONE_CONFIG:
        return jsonify({"error": f"Milestone '{milestone}' is not recognized in MILESTONE_CONFIG."}), 400

    # Check if this milestone is enabled
    if not MILESTONE_CONFIG[milestone]["enabled"]:
        return jsonify({"error": f"Milestone '{milestone}' is disabled."}), 400

    # If extended is true, check if milestone has extended enabled
    if extended_param == "true":
        if MILESTONE_CONFIG[milestone].get("extended_enabled", False):
            milestone = f"{milestone}_extended"
        else:
            return jsonify({"error": f"Extended version of milestone '{milestone}' is not enabled."}), 400

    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_folder_name = f"lstore_eval_{timestamp}_{unique_id}"
    extract_path = os.path.join(SUBMISSIONS_DIR, unique_folder_name)
    os.makedirs(extract_path, exist_ok=True)

    # 1) Extract or clone user-submitted code
    if file and file.filename.endswith('.zip'):
        lstore_path = os.path.join(extract_path, 'lstore')
        os.makedirs(lstore_path, exist_ok=True)
        with zipfile.ZipFile(BytesIO(file.read())) as zip_file:
            top_level_dirs = set()
            for name in zip_file.namelist():
                parts = name.split('/')
                if len(parts) > 0 and parts[0]:
                    top_level_dirs.add(parts[0])
            if len(top_level_dirs) == 1:
                folder_name = list(top_level_dirs)[0]
                for member in zip_file.namelist():
                    member_path = os.path.relpath(member, folder_name)
                    if member_path == '.':
                        continue
                    target_path = os.path.join(lstore_path, member_path)
                    if member.endswith('/'):
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with open(target_path, 'wb') as f:
                            f.write(zip_file.read(member))
            else:
                zip_file.extractall(lstore_path)
    elif github_repo:
        # Clone directly into extract_path (no "repo" subfolder)
        lstore_path = extract_path
        try:
            subprocess.check_call(["git", "clone", github_repo, lstore_path])
        except Exception as e:
            return jsonify({"error": f"Git clone failed: {e}"}), 400
    else:
        return jsonify({"error": "No valid file or GitHub repository provided."}), 400

    # 2) Run milestone tests
    results, m_tests, m_count, total = milestone_tests(
        extract_path,
        lstore_path,
        milestone,
        timeout_val
    )

    # 3) Optionally run AI checks
    if ai_param:
        ai_results = asyncio.run(run_ai_tests(lstore_path, ai_timeout=ai_timeout_val))
    else:
        ai_results = {}

    # 4) Extract commit stats
    commit_stats = extract_commit_stats(extract_path)
    if github_repo:
        repo_stats = extract_commit_stats(lstore_path)
        # store the raw lines for debugging
        commit_stats["github_repo_commits"] = repo_stats.get("local_git_commits", [])
        # Merge contributors from local & github
        merged = merge_contributors(
            commit_stats.get("contributors", []),
            repo_stats.get("contributors", [])
        )
        commit_stats["contributors"] = merged

    # Stop timing now
    overall_end = time.time()
    total_subprocess_time = round(overall_end - overall_start, 3)

    # 5) Prepare final result
    final_output = {
        "submission_name": submission_name,
        "milestone": milestone,
        "tests": m_tests,
        "passed": m_count,
        "total_tests": total,
        "performance_results": results,  # e.g. insert_time, update_time, etc.
        "ai_tests": ai_results,
        "commit_stats": commit_stats,
        # Add total time spent in the entire pipeline
        "subprocess_total_time": total_subprocess_time
    }

    # 6) Store on ResilientDB
    tx_id = store_on_resilientdb(final_output)
    if tx_id:
        final_output["resilientdb_tx_id"] = tx_id

    # 7) Insert into Mongo
    insert_result = db["new_lstore"].insert_one(dict(final_output))
    final_output["_id"] = str(insert_result.inserted_id)

    return jsonify(final_output)


if __name__ == "__main__":
    app.run(port="7200", debug=True)
