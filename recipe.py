from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and inspect the data
df = pd.read_csv('Food_Recipe.csv')
print("Columns in the CSV:", df.columns)  # Debug: Print columns to check names

# Rename column if necessary
if 'ingredients_name' in df.columns:
    df = df.rename(columns={'ingredients_name': 'ingredients'})
elif 'Ingredients' in df.columns:  # Adjust based on your actual column name
    df = df.rename(columns={'Ingredients': 'ingredients'})

# Clean and preprocess the data
df = df.drop_duplicates('name')
df = df.dropna()
df = df.reset_index(drop=True)
df['ingredients'] = df['ingredients'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
df['ingredients'] = df['ingredients'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in ['chopped', 'fresh', 'sliced']])
)

# Updated allowed cuisines list (same as before)
allowed_cuisines = [
    'Mexican', 'North Indian Recipes', 'South Karnataka',
    'Italian Recipes', 'Rajasthani', 'Bihari', 'Indian',
    'Kerala Recipes', 'Continental', 'French', 'Bengali Recipes',
    'Coastal Karnataka', 'Tamil Nadu', 'Sri Lankan', 'Goan Recipes', 'Karnataka',
    'South Indian Recipes', 'Maharashtrian Recipes',
    'Gujarati Recipesï»¿',  'Japanese', 'Sichuan', 'Chinese',
    'Arab', 'Asian', 'Mughlai', 'Chettinad', 'Indo Chinese', 'Awadhi',
    'Mediterranean', 'Punjabi',  'Korean',
    'Kashmiri',
    'North East India Recipes',  'Uttar Pradesh', 'European', 'Lucknowi',
    'Nepalese', 'Vietnamese', 'Haryana', 'Indonesian', 'Sri Lanka',
    'Sindhi',
    'Hunan', 'Jharkhand', 'Assamese',  'Uttarakhand - North Kumaon',

]

# Function to recommend recipes based on ingredients and allowed cuisines
def get_recipe_recommendations(ingredients_list, allowed_cuisines, selected_cuisine, df):
    ingredients_list = [ingredient.lower().strip() for ingredient in ingredients_list]
    cv_ingredients = CountVectorizer()
    count_matrix = cv_ingredients.fit_transform(df['ingredients'])
    ingredients_vector = cv_ingredients.transform([' '.join(ingredients_list)])
    cosine_similarities = cosine_similarity(ingredients_vector, count_matrix).flatten()
    sorted_indices = cosine_similarities.argsort()[::-1]

    recommended_recipes = []
    for i in sorted_indices:
        # Filter by selected cuisine if specified
        if selected_cuisine and df.iloc[i]['cuisine'] == selected_cuisine:
            recipe_ingredients = set(df.iloc[i]['ingredients'].split())
            if all(ingredient in recipe_ingredients for ingredient in ingredients_list):
                recommended_recipes.append(df.iloc[i])
        elif not selected_cuisine:  # No cuisine selected, return all matching recipes
            recipe_ingredients = set(df.iloc[i]['ingredients'].split())
            if all(ingredient in recipe_ingredients for ingredient in ingredients_list):
                recommended_recipes.append(df.iloc[i])
        if len(recommended_recipes) >= 10:
            break
    return recommended_recipes

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        ingredients_input = request.form.get('ingredients')
        user_ingredients = [ingredient.strip() for ingredient in ingredients_input.split(',')]
        selected_cuisine = request.form.get('cuisine')  # Get selected cuisine from the form
        recommendations = get_recipe_recommendations(user_ingredients, allowed_cuisines, selected_cuisine, df)
        return render_template('suggestions.html', recipes=recommendations)
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
