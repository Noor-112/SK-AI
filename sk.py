import pandas as pd
import re
import ast
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

print("Loading AI Dataset...")
try:
    df = pd.read_csv('RAW_recipes after cleaning.csv')
    
    columns_to_keep = [
        'name', 'ingredients', 'minutes', 'n_ingredients', 'n_steps', 'tags',
        'calories', 'fat', 'carbs', 'protein', 'sugar', 'fiber', 'sodium'
    ]
    
    for col in columns_to_keep:
        if col not in df.columns:
            df[col] = "[]" if col in ['tags', 'ingredients'] else 0.0
            
    df['name'] = df['name'].fillna('')
    df['ingredients'] = df['ingredients'].fillna('[]')
    df['tags'] = df['tags'].fillna('[]')
    
    numeric_cols = ['minutes', 'n_ingredients', 'n_steps', 'calories', 'fat', 'carbs', 'protein', 'sugar', 'fiber', 'sodium']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    df = df[columns_to_keep].reset_index(drop=True)
    
    def simple_clean(text):
        if isinstance(text, str):
            text = re.sub(r'[\[\]\'\"]', '', text)
            text = ' '.join(text.split())
            return text.lower()
        return ''
    
    df['ingredients_clean'] = df['ingredients'].apply(simple_clean)

    print(f"Loaded {len(df)} recipes successfully! Vectorized Turbo Mode Activated ")
except Exception as e:
    print(f"Error Loading CSV: {e}")
    df = pd.DataFrame()

class ProfilePreferences(BaseModel):
    diets: Optional[List[str]] = None       
    allergens: Optional[List[str]] = None   

class AdvancedFilters(BaseModel):
    skill_level: Optional[List[str]] = None 
    diet: Optional[List[str]] = None        
    meal: Optional[List[str]] = None        
    time: Optional[List[str]] = None        
    cuisines: Optional[List[str]] = None    
    category: Optional[List[str]] = None    

class RecipeRequest(BaseModel):
    recipe_name: Optional[str] = None
    exact_match: bool = False
    ingredients: Optional[List[str]] = None
    top_n: int = 5
    filters: Optional[AdvancedFilters] = None
    profile_prefs: Optional[ProfilePreferences] = None

ALLERGEN_KEYWORDS = {
    "Dairy": ["milk", "cheese", "butter", "cream", "yogurt", "lactose", "whey"],
    "Gluten": ["wheat", "barley", "rye", "flour", "bread", "pasta"],
    "Garlic": ["garlic"],
    "Onion": ["onion"],
    "Egg": ["egg", "yolk", "albumen"],
    "Soy": ["soy", "tofu", "edamame"],
    "Fish": ["fish", "salmon", "tuna", "cod", "tilapia", "sardine"],
    "Seafood": ["shrimp", "crab", "lobster", "mussel", "clam", "scallop", "oyster"],
    "Nuts": ["nut", "peanut", "almond", "cashew", "walnut", "pecan", "pistachio"],
    "Mushroom": ["mushroom"]
}

@app.post("/recommend")
def get_recommendations(request: RecipeRequest):
    top_n = request.top_n
    filters = request.filters
    prefs = request.profile_prefs
    
    search_term = request.recipe_name.lower().strip() if request.recipe_name else None
    user_set = set([ing.lower().strip() for ing in request.ingredients]) if request.ingredients else set()
    user_set_len = len(user_set)
    
    # هناخد نسخة من الداتا نشتغل عليها فلترة سريعة
    working_df = df.copy()

    # ==========================================
    # 1. الفلترة (Pandas Vectorization)
    # ==========================================

    # --- فلاتر البحث بالاسم ---
    if search_term:
        if request.exact_match:
            working_df = working_df[working_df['name'].astype(str).str.lower() == search_term]
        else:
            working_df = working_df[working_df['name'].astype(str).str.contains(search_term, case=False, na=False, regex=False)]

    if prefs:
        if prefs.allergens and "None" not in prefs.allergens:
            for allergen in prefs.allergens:
                kws = ALLERGEN_KEYWORDS.get(allergen, [allergen.lower()])
                pattern = '|'.join(kws)
                working_df = working_df[~working_df['ingredients_clean'].str.contains(pattern, case=False, na=False, regex=True)]

        if prefs.diets and "None" not in prefs.diets:
            for d in prefs.diets:
                if d.lower() in ["non-veg", "non-vegetarian"]:
                    working_df = working_df[~working_df['tags'].astype(str).str.contains('vegetarian|vegan', case=False, na=False, regex=True)]
                else:
                    d_term = d.lower().replace(" ", "-")
                    working_df = working_df[working_df['tags'].astype(str).str.contains(f'{d_term}|{d.lower()}', case=False, na=False, regex=True)]

    # --- فلاتر الشاشات المتقدمة ---
    if filters:
        if filters.time:
            mask = pd.Series(False, index=working_df.index)
            for t in filters.time:
                if t == "5 - 10 min": mask |= (working_df['minutes'] >= 5) & (working_df['minutes'] <= 10)
                elif t == "10 - 20 min": mask |= (working_df['minutes'] > 10) & (working_df['minutes'] <= 20)
                elif t == "20 - 30 min": mask |= (working_df['minutes'] > 20) & (working_df['minutes'] <= 30)
                elif t == "30 - 45 min": mask |= (working_df['minutes'] > 30) & (working_df['minutes'] <= 45)
                elif t == "45 - 60 min": mask |= (working_df['minutes'] > 45) & (working_df['minutes'] <= 60)
                elif t == "> 1 hr": mask |= (working_df['minutes'] > 60)
            working_df = working_df[mask]

        if filters.meal:
            mask = pd.Series(False, index=working_df.index)
            for m in filters.meal:
                m_term = m.lower().replace("appetiser", "appetizer").replace(" & ", " ")
                m_search = m_term[:-1] if m_term.endswith('s') else m_term
                mask |= working_df['tags'].astype(str).str.contains(m_search, case=False, na=False, regex=False)
            working_df = working_df[mask]

        if filters.diet:
            mask = pd.Series(False, index=working_df.index)
            for d in filters.diet:
                if d.lower() == "non-veg":
                    mask |= working_df['tags'].astype(str).str.contains('meat|poultry|fish', case=False, na=False, regex=True)
                else:
                    d_term = d.lower().replace(" ", "-")
                    mask |= working_df['tags'].astype(str).str.contains(f'{d_term}|{d.lower()}', case=False, na=False, regex=True)
            working_df = working_df[mask]

        if filters.cuisines:
            mask = pd.Series(False, index=working_df.index)
            for c in filters.cuisines:
                mask |= working_df['tags'].astype(str).str.contains(c.lower(), case=False, na=False, regex=False)
            working_df = working_df[mask]

        if filters.category:
            mask = pd.Series(False, index=working_df.index)
            for c in filters.category:
                c_term = c.lower().replace(" & ", " ")
                c_search = c_term[:-1] if c_term.endswith('s') else c_term
                mask |= working_df['tags'].astype(str).str.contains(c_search, case=False, na=False, regex=False)
            working_df = working_df[mask]

        if filters.skill_level:
            mask = pd.Series(False, index=working_df.index)
            tags_str = working_df['tags'].astype(str).str.lower()
            for diff in filters.skill_level:
                d = diff.lower()
                if d == 'easy':
                    mask |= (working_df['minutes'] <= 30) | tags_str.str.contains('easy', regex=False)
                elif d == 'medium':
                    mask |= (working_df['minutes'] > 30) & (working_df['minutes'] <= 60)
                elif d == 'advanced':
                    mask |= (working_df['minutes'] > 60) | (working_df['n_steps'] > 12)
            working_df = working_df[mask]

    # ==========================================
    # 2. حساب السكور وترتيب النتائج
    # ==========================================
    
    # عشان منعطلش السيرفر أبداً، لو الداتا لسه كبيرة جداً بعد الفلترة، هناخد أول 2000 وصفة بس نطبق عليهم لوجيك السكور
    working_df = working_df.head(2000)
    
    best_matches = []
    if not working_df.empty:
        for idx, row in working_df.iterrows():
            recipe_text = str(row['ingredients_clean']).lower()
            recipe_name_db = str(row['name']).lower().strip()

            # وضع "التصفح": لو مفيش بحث بالاسم أو المكونات 
            if not search_term and user_set_len == 0:
                best_matches.append({
                    'name': row['name'],
                    'score': 1.0, # بنديله سكور كامل لأنه طابق الفلاتر
                    'match_pct': 100.0,
                    'common_count': 1
                })
                continue

            # وضع "البحث الاحترافي"
            name_score = 0.0
            ing_score = 0.0
            is_valid_match = False
            common_len = 1

            if search_term:
                if request.exact_match:
                    name_score = 1.0
                    is_valid_match = True
                else:
                    name_score = min(0.5 + ((len(search_term) / len(recipe_name_db)) * 0.4), 0.95) if len(recipe_name_db) > 0 else 0
                    if search_term == recipe_name_db: name_score = 1.0
                    is_valid_match = True

            if user_set_len > 0:
                recipe_ings = set(recipe_text.split())
                common = recipe_ings & user_set
                common_len = len(common)
                match_pct = common_len / user_set_len
                
                if match_pct > 0:
                    ing_score = min(0.3 + (match_pct * 0.7), 0.95)
                    is_valid_match = True

            if is_valid_match:
                final_score = 0.0
                if search_term and user_set_len > 0:
                    if name_score == 0.0 or ing_score == 0.0: continue 
                    final_score = (name_score * 0.4) + (ing_score * 0.6)
                elif search_term and user_set_len == 0:
                    final_score = name_score
                elif user_set_len > 0 and not search_term:
                    final_score = ing_score
                
                if final_score > 0:
                    best_matches.append({
                        'name': row['name'],
                        'score': round(final_score, 3), 
                        'match_pct': round((final_score * 100), 1),
                        'common_count': common_len if user_set_len > 0 else 1
                    })
                
        if best_matches:
            best_matches.sort(key=lambda x: (x['score'], x['common_count'], x['match_pct']), reverse=True)
            results = [{"name": m['name'], "score": m['score']} for m in best_matches[:top_n]]
            return {"recommendations": results}
            
    return {"recommendations": [{"name": "No recipes found matching your criteria.", "score": 0.0}]}
