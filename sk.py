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
    
    columns_to_keep = ['name', 'ingredients', 'nutrition', 'minutes', 'n_ingredients', 'n_steps', 'tags']
    for col in columns_to_keep:
        if col not in df.columns:
            df[col] = "[]" if col in ['nutrition', 'tags'] else 0
            
    df = df[columns_to_keep].dropna().reset_index(drop=True)
    
    def simple_clean(text):
        if isinstance(text, str):
            text = re.sub(r'[\[\]\'\"]', '', text)
            text = ' '.join(text.split())
            return text.lower()
        return ''
    
    df['ingredients_clean'] = df['ingredients'].apply(simple_clean)

    def parse_nutrition(nutr_str):
        try:
            val = ast.literal_eval(nutr_str)
            return {
                'fat': float(val[1]),
                'sugar': float(val[2]),
                'sodium': float(val[3]),
                'protein': float(val[4]),
                'carbs': float(val[6])
            }
        except:
            return {'fat': 0, 'sugar': 0, 'sodium': 0, 'protein': 0, 'carbs': 0}
    
    print("Extracting health values...")
    nutr_df = pd.DataFrame(list(df['nutrition'].apply(parse_nutrition)))
    df = pd.concat([df, nutr_df], axis=1)

    for col in ['carbs', 'fat', 'protein', 'sugar', 'fiber', 'sodium']:
        if col not in df.columns:
            df[col] = 0.0

    print(f"Loaded {len(df)} recipes successfully!")
except Exception as e:
    print(f"Error Loading CSV: {e}")
    df = pd.DataFrame()

class ProfilePreferences(BaseModel):
    diets: Optional[List[str]] = None
    allergens: Optional[List[str]] = None


class AdvancedFilters(BaseModel):
    min_carbs: float = 0
    max_carbs: float = 10000
    min_fat: float = 0
    max_fat: float = 10000
    min_protein: float = 0
    max_protein: float = 10000
    min_sugar: float = 0
    max_sugar: float = 10000
    min_fiber: float = 0
    max_fiber: float = 10000
    min_sodium: float = 0
    max_sodium: float = 10000
    min_time: int = 0
    max_time: int = 10000
    min_ingredients: int = 0
    max_ingredients: int = 10000
    meal: Optional[List[str]] = None
    diet: Optional[List[str]] = None
    skill_level: Optional[List[str]] = None

class RecipeRequest(BaseModel):
    recipe_name: Optional[str] = None
    ingredients: Optional[List[str]] = None
    top_n: int = 5
    filters: Optional[AdvancedFilters] = None
    profile_prefs: Optional[ProfilePreferences] = None

@app.post("/recommend")
def get_recommendations(request: RecipeRequest):
    top_n = request.top_n
    filters = request.filters
    prefs = request.profile_prefs
            
    best_matches = []
    if not df.empty:
        for idx, row in df.iterrows():
            
        
            if prefs:
               
                if prefs.allergens and len(prefs.allergens) > 0:
                    found_allergen = False
                    recipe_text = str(row['ingredients_clean']).lower()
                    for allergen in prefs.allergens:
                        if allergen.lower().strip() in recipe_text:
                            found_allergen = True
                            break
                    if found_allergen: continue

               
                if prefs.diets and len(prefs.diets) > 0:
                    row_tags = str(row['tags']).lower()
                    diet_match = True
                    for d in prefs.diets:
                        d_term = d.lower().replace(" ", "-")
                        if d_term not in row_tags and d.lower() not in row_tags:
                            diet_match = False
                            break
                    if not diet_match: continue

            if filters:
                if not (filters.min_carbs <= row['carbs'] <= filters.max_carbs and
                        filters.min_fat <= row['fat'] <= filters.max_fat and
                        filters.min_protein <= row['protein'] <= filters.max_protein and
                        filters.min_sugar <= row['sugar'] <= filters.max_sugar and
                        filters.min_sodium <= row['sodium'] <= filters.max_sodium):
                    continue
                
                if not (filters.min_time <= row['minutes'] <= filters.max_time):
                    continue
                if not (filters.min_ingredients <= row['n_ingredients'] <= filters.max_ingredients):
                    continue
                    
                row_tags_str = str(row['tags']).lower()
                
                if filters.meal and len(filters.meal) > 0:
                    if not any(m.lower() in row_tags_str for m in filters.meal):
                        continue
                
                if filters.diet and len(filters.diet) > 0:
                    diet_matched = True
                    for d in filters.diet:
                        diet_term = d.lower().replace(" ", "-") 
                        if diet_term not in row_tags_str and d.lower() not in row_tags_str:
                            diet_matched = False
                            break
                    if not diet_matched: continue
                        
                if filters.skill_level and len(filters.skill_level) > 0:
                    diff_matched = False
                    minutes = row['minutes']
                    steps = row.get('n_steps', 5)
                    for diff in filters.skill_level:
                        d = diff.lower()
                        if d == 'easy' and (minutes <= 30 or 'easy' in row_tags_str):
                            diff_matched = True
                        elif d == 'medium' and (30 < minutes <= 60):
                            diff_matched = True
                        elif d == 'advanced' and (minutes > 60 or steps > 12): 
                            diff_matched = True
                    if not diff_matched: continue

       
            if request.recipe_name:
                search_term = request.recipe_name.lower()
                if search_term in str(row['name']).lower():
                    score = 1.0 if search_term == str(row['name']).lower() else 0.8
                    best_matches.append({'name': row['name'], 'score': score, 'match_pct': 100, 'common_count': 1})
                    
            elif request.ingredients and len(request.ingredients) > 0:
                user_ingredients = [ing.lower() for ing in request.ingredients]
                recipe_ings = set(row['ingredients_clean'].split())
                user_set = set(user_ingredients)
                common = recipe_ings & user_set
                match_pct = len(common) / len(user_set) if len(user_set) > 0 else 0
                
                if len(common) > 0:
                    score = min(0.3 + (match_pct * 0.7), 0.95)
                    best_matches.append({
                        'name': row['name'],
                        'score': score,
                        'match_pct': match_pct * 100,
                        'common_count': len(common)
                    })
                
        if best_matches:
            best_matches.sort(key=lambda x: (x['common_count'], x['match_pct'], x['score']), reverse=True)
            results = [{"name": m['name'], "score": m['score']} for m in best_matches[:top_n]]
            return {"recommendations": results}
            
    return {"recommendations": [{"name": "No recipes found matching your criteria.", "score": 0.0}]}