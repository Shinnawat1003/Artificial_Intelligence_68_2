# Pattern Recognition - Probability Theory with User Input

# 1) PRIOR PROBABILITIES
print("=== PRIOR PROBABILITIES ===")
p_red = float(input("P(Box = red): "))      # example: 0.40
p_blue = float(input("P(Box = blue): "))    # example: 0.60

# 2) NUMBER OF FRUITS IN EACH BOX
print("\n=== NUMBER OF FRUITS IN EACH BOX ===")
red_apple  = int(input("Red box: number of apples: "))   # example: 2
red_orange = int(input("Red box: number of oranges: "))  # example: 6

blue_apple  = int(input("Blue box: number of apples: "))   # example: 3
blue_orange = int(input("Blue box: number of oranges: "))  # example: 1

# Total fruits
total_red = red_apple + red_orange
total_blue = blue_apple + blue_orange

# 3) LIKELIHOOD: P(F | Box)
p_a_given_red = red_apple / total_red
p_o_given_red = red_orange / total_red

p_a_given_blue = blue_apple / total_blue
p_o_given_blue = blue_orange / total_blue

# 4) MARGINAL PROBABILITY: P(F)
p_a = p_a_given_red * p_red + p_a_given_blue * p_blue
p_o = p_o_given_red * p_red + p_o_given_blue * p_blue

# 5) POSTERIOR: P(red | orange)
p_red_given_o = p_o_given_red * p_red / p_o

# 6) PRINT RESULTS
print("\n=== Given information ===")
print(f"P(Box = red)  = {p_red:.2f}")
print(f"P(Box = blue) = {p_blue:.2f}")
print(f"Red box  : {red_apple} apples, {red_orange} oranges")
print(f"Blue box : {blue_apple} apples, {blue_orange} oranges")
print()

print("=== Marginal probabilities of fruit ===")
print(f"P(F = apple)  = {p_a:.4f}  ({p_a*100:.2f}%)")
print(f"P(F = orange) = {p_o:.4f}  ({p_o*100:.2f}%)")
print()

print("=== Posterior probability ===")
print(f"P(Box = red | F = orange) = {p_red_given_o:.4f}  ({p_red_given_o*100:.2f}%)")
