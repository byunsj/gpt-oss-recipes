#!/usr/bin/env python3
"""
사용자의 실제 데이터를 Hugging Face 형식으로 변환하는 스크립트
"""
import os
import json
import ast
import pandas as pd
from datasets import Dataset

# 사용자의 원래 프롬프트 설정
SYSTEM_PROMPT = (
    "You are an expert crisis annotator. "
    "Given a social media post, label all applicable crisis categories. "
    'Respond ONLY with a JSON object: {"labels": ["..."]}.'
)

def read_user_prompt(path: str) -> str:
    """사용자의 프롬프트 파일 읽기"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        # 기본 프롬프트 제공
        return """Identify crisis categories from the following social media post. 
Consider categories like: depression, anxiety, suicide_ideation, self_harm, substance_abuse, eating_disorder, trauma, no_crisis.
Respond with a JSON object containing the applicable labels."""

def render_user_message_train(title: str, text: str, user_prompt_template: str) -> str:
    """훈련 데이터용 사용자 메시지 생성"""
    return f"{user_prompt_template}\n\nTitle: {title}\nPost:\n{text}\n"

def render_user_message_dev(question_text: str, user_prompt_template: str) -> str:
    """개발 데이터용 사용자 메시지 생성"""
    return f"{user_prompt_template}\n\nPost:\n{question_text}\n"

def normalize_labels(raw):
    """라벨을 표준화"""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "labels" in obj:
                return obj["labels"]
            if isinstance(obj, list):
                return obj
        except:
            pass
        try:
            lit = ast.literal_eval(s)
            if isinstance(lit, dict) and "labels" in lit:
                return lit["labels"]
            if isinstance(lit, list):
                return lit
        except:
            pass
    return []

def ensure_json_labels(raw):
    """라벨을 JSON 형식으로 변환"""
    labels = normalize_labels(raw)
    return json.dumps({"labels": labels}, ensure_ascii=False)

def convert_to_messages_format(df, is_train=True, user_prompt_template=""):
    """DataFrame을 messages 형식으로 변환"""
    messages_data = []
    
    for _, row in df.iterrows():
        # 사용자 메시지 생성
        if is_train and "title" in row and "text" in row:
            user_content = render_user_message_train(
                str(row.get("title", "")),
                str(row.get("text", "")),
                user_prompt_template
            )
            labels_raw = row.get("voted_labels", "[]")
        else:
            user_content = render_user_message_dev(
                str(row.get("question_text", "")),
                user_prompt_template
            )
            labels_raw = row.get("final_labels", "[]")
        
        # 어시스턴트 응답 생성
        assistant_content = ensure_json_labels(labels_raw)
        
        # 메시지 형식으로 구성
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        
        messages_data.append({
            "messages": messages,
            "labels": normalize_labels(labels_raw)
        })
    
    return messages_data

def main():
    """메인 함수"""
    print("🔄 Converting user data to Hugging Face format...")
    
    # 사용자 프롬프트 읽기
    user_prompt_template = read_user_prompt("./prompt_short.txt")
    print(f"📝 User prompt template loaded ({len(user_prompt_template)} chars)")
    
    # CSV 파일들이 존재하는지 확인
    train_file = "reddit_train_voted_2plus_agrees.csv"
    dev_file = "devset_420_final.csv"
    
    if not os.path.exists(train_file):
        print(f"❌ Training file not found: {train_file}")
        print("Creating sample data for demonstration...")
        create_sample_data()
        return
    
    if not os.path.exists(dev_file):
        print(f"❌ Dev file not found: {dev_file}")
        print("Creating sample data for demonstration...")
        create_sample_data()
        return
    
    # 데이터 로드
    print(f"📂 Loading {train_file}...")
    train_df = pd.read_csv(train_file, encoding="utf-8")
    print(f"📂 Loading {dev_file}...")
    dev_df = pd.read_csv(dev_file, encoding="utf-8")
    
    print(f"✅ Train shape: {train_df.shape}")
    print(f"✅ Dev shape: {dev_df.shape}")
    
    # 메시지 형식으로 변환
    print("🔄 Converting to messages format...")
    train_messages = convert_to_messages_format(train_df, is_train=True, user_prompt_template=user_prompt_template)
    dev_messages = convert_to_messages_format(dev_df, is_train=False, user_prompt_template=user_prompt_template)
    
    # Dataset 생성
    train_dataset = Dataset.from_list(train_messages)
    dev_dataset = Dataset.from_list(dev_messages)
    
    # 저장
    output_dir = "./user_crisis_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset.save_to_disk(f"{output_dir}/train")
    dev_dataset.save_to_disk(f"{output_dir}/test")
    
    print(f"✅ Dataset saved to {output_dir}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(dev_dataset)}")
    
    # 샘플 확인
    print("\n📊 Sample data:")
    sample = train_dataset[0]
    print(f"   Messages: {len(sample['messages'])}")
    for i, msg in enumerate(sample['messages']):
        role = msg['role']
        content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
        print(f"   {i+1}. {role}: {content_preview}")

def create_sample_data():
    """샘플 데이터 생성 (실제 파일이 없을 때)"""
    print("🔄 Creating sample crisis dataset...")
    
    # 샘플 훈련 데이터
    train_data = [
        {
            "title": "Feeling hopeless",
            "text": "I've been feeling really down lately and don't see the point in anything. Everything feels meaningless.",
            "voted_labels": '["depression", "suicide_ideation"]'
        },
        {
            "title": "Anxiety about work",
            "text": "I can't stop worrying about my job performance. I'm having panic attacks every morning.",
            "voted_labels": '["anxiety"]'
        },
        {
            "title": "Good day today",
            "text": "Had a great day with friends, feeling grateful for the support system I have.",
            "voted_labels": '["no_crisis"]'
        }
    ]
    
    # 샘플 개발 데이터
    dev_data = [
        {
            "question_text": "I've been cutting myself when I feel overwhelmed. I know it's not healthy but I don't know how to stop.",
            "final_labels": '["self_harm", "depression"]'
        },
        {
            "question_text": "Just got promoted at work! Celebrating with my family tonight.",
            "final_labels": '["no_crisis"]'
        }
    ]
    
    # DataFrame 생성
    train_df = pd.DataFrame(train_data)
    dev_df = pd.DataFrame(dev_data)
    
    # 기본 프롬프트 생성
    user_prompt_template = """Identify crisis categories from the following social media post. 
Consider categories like: depression, anxiety, suicide_ideation, self_harm, substance_abuse, eating_disorder, trauma, no_crisis.
Respond with a JSON object containing the applicable labels."""
    
    # 변환 및 저장
    train_messages = convert_to_messages_format(train_df, is_train=True, user_prompt_template=user_prompt_template)
    dev_messages = convert_to_messages_format(dev_df, is_train=False, user_prompt_template=user_prompt_template)
    
    train_dataset = Dataset.from_list(train_messages)
    dev_dataset = Dataset.from_list(dev_messages)
    
    output_dir = "./user_crisis_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset.save_to_disk(f"{output_dir}/train")
    dev_dataset.save_to_disk(f"{output_dir}/test")
    
    print(f"✅ Sample dataset created at {output_dir}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(dev_dataset)}")

if __name__ == "__main__":
    main()