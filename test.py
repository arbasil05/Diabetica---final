def inches_to_meters(inches):
    meters = inches * 0.0254
    return meters

inches = float(input("Enter length in inches: "))
meters = inches_to_meters(inches)
print(f"{inches} inches is equal to {meters:.2f} meters.")
