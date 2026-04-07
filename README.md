# ML_Project

This Streamlit application, "Checkmate Catalyst," is a machine learning tool designed to analyze chess endgames using a Decision Tree algorithm.

## **1. What is it used for?**
The primary purpose of this tool is **Classification**. 

In the context of chess, it takes a dataset of endgame positions (specifically the "King+Rook vs. King+Pawn" dataset from OpenML) and predicts whether the position results in a win for white. It serves two roles:
* **Strategic Analysis:** Identifying which squares on the board are the most "informative" or critical for determining the outcome of a match.
* **Educational Tool:** Demonstrating how a machine "thinks" by breaking down complex game states into a series of simple "Yes/No" strategic questions.

## **2. How does it work? (The Engine)**

The app follows a standard machine learning pipeline, stylized as a chess engine:

### **A. Data Processing**
The app fetches a dataset where every row represents a chess board configuration. Since machine learning models can't "see" letters like 'a1' or 'h8', it uses **One-Hot Encoding** (`pd.get_dummies`). This converts board positions into numerical values that the computer can process.

### **B. The Decision Tree (The Logic)**
The "brain" of the app is the **Decision Tree Classifier**. You can think of this as a flow chart of strategic moves:
1.  **Binary Splitting:** The model looks at all squares and pieces. It asks: *"Is there a Rook on f7?"*
2.  **Information Gain:** It chooses the question that best separates "Wins" from "Draws." This is what the **Gini** or **Entropy** settings in your sidebar control—they are mathematical formulas used to measure the "purity" of the resulting groups.
3.  **Recursive Branching:** It repeats this process until it reaches the **Engine Depth** you set. A higher depth means a more complex, granular analysis of the game.

### **C. The "Chess.com" Interface**
To make the data digestible, the app translates math into visuals:
* **Evaluation Bar:** The vertical white bar shows the model's **Accuracy**. If it's $90\%$ accurate, the bar fills up, mirroring the "Advantage" bar seen during live grandmaster games.
* **Analysis Board:** Instead of a chess board, it displays the **Tree Plot**. This shows the specific logic gates the model used to reach its conclusion.
* **Top Informative Squares:** This uses **Feature Importance** to show you which areas of the board had the biggest impact on the final prediction.

## **3. Key Technical Concepts**
* **Recursive Binary Splitting:** The process of repeatedly dividing the data into two groups to minimize uncertainty.
* **Heuristics (Gini/Entropy):** The criteria used to decide which "square" or "move" provides the most information.
* **Overfitting:** If you set the **Engine Depth** too high (e.g., 20), the model might "memorize" the specific games in the dataset rather than learning general chess principles.
