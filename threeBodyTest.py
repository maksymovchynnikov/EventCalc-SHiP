# decay_simulation.py

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from funcs.ThreeBodyDecay import decay_products  # Updated absolute import
import sys

def main():
    """
    Main function to generate three-body decay events and plot a histogram
    of the energy of the first decay product.
    """
    # -----------------------------
    # 1. Define Decay Parameters
    # -----------------------------

    # Decaying particle properties
    decaying_particle_mass = 1.0  # Mass of the decaying particle (GeV)

    # Decay products: PDG IDs, Masses (GeV), Charges, Stabilities
    #pdgs = [211., -211., 111.]            # PDG codes for pi+, pi-, pi0
    #masses = [1.0, 0.139, 0.139, 0.135]   # Masses: [Decaying Particle, pi+, pi-, pi0] (GeV)
    #charges = [0., 1., -1., 0.]            # Charges: [Decaying Particle, pi+, pi-, pi0]
    #stabilities = [1., 1., 1., 0.]         # Stabilities: [Decaying Particle, pi+, pi-, pi0]
    pdgs = [22., -11., 11.]
    masses = [1.0, 0.0, 0.000511, 0.000511]
    charges = [0., 0., 1., -1.] 
    stabilities = [1., 1., 1., 1.]

    # -----------------------------
    # 2. Compile Matrix Element
    # -----------------------------

    # Matrix element expression as a string
    matrix_element_str = (
        "(4*mLLP**2*(0.0003733010410000002 - 0.019321000000000005*E1**2 - "
        "0.019321000000000005*(E1 + E3 - 1.*mLLP)**2 + E1*(E1 + E3 - mLLP)*"
        "(0.019321000000000005 + 2*E3*mLLP - mLLP**2) - 1.*"
        "(0.009660500000000002 + E3*mLLP - 0.5*mLLP**2)**2)*"
        "(16*E1**2*E3**2*mLLP**4 + 16*(E1 - E3)**2*(E1 + E3 - mLLP)**2*mLLP**4 + "
        "(0.35089324604100003 - 1.162608*mLLP**2 + mLLP**4)*(0.35089324604100003 - "
        "1.162608*mLLP**2 + 16*E3**2*mLLP**2 + mLLP**4 - "
        "8.*E3*mLLP*(-0.581304 + mLLP**2)) + "
        "8*(E1 - E3)*(E1 + E3 - mLLP)*mLLP**2*"
        "(0.32493543479100007 - 1.162608*mLLP**2 + mLLP**4 - "
        "4.*E3*mLLP*(-0.581304 - 1.*E1*mLLP + mLLP**2)) - "
        "32.*E1*E3*mLLP**2*(-0.08123385869775002 + 0.290652*mLLP**2 - "
        "0.25*mLLP**4 + E3*(-0.581304*mLLP + mLLP**3)))) / "
        "((0.35089324604100003 - 1.162608*mLLP**2 + 4*E1**2*mLLP**2 + "
        "mLLP**4 - 4.*E1*mLLP*(-0.581304 + mLLP**2)) * "
        "(0.35089324604100003 - 1.162608*mLLP**2 + 4*E3**2*mLLP**2 + "
        "mLLP**4 - 4.*E3*mLLP*(-0.581304 + mLLP**2)) * "
        "(0.35089324604100003 - 1.162608*mLLP**2 + 4*(E1 + E3 - mLLP)**2*mLLP**2 + "
        "mLLP**4 - 4.*mLLP*(-1.*E1 - 1.*E3 + mLLP)*(-0.581304 + mLLP**2)))"
    )
    matrix_element_str = ("(-32*(4*E1**3*mLLP - mLLP**2*(-2*E3 + mLLP)**2 + E1**2*(-1.***-6 + 8*E3*mLLP - 6*mLLP**2) + 4*E1*mLLP*(2*E3**2 - 3*E3*mLLP + mLLP**2))*Pi)/(-2*E1 + mLLP)**2")

    # Define symbols for SymPy
    mLLP, E1, E3 = sp.symbols('mLLP E1 E3')
    local_dict = {'mLLP': mLLP, 'E1': E1, 'E3': E3}

    # Replace '***' with 'e' to handle scientific notation (if present)
    matrix_element_str_corrected = matrix_element_str.replace('***', 'e')

    # Parse the corrected expression with SymPy
    try:
        expr = sp.sympify(matrix_element_str_corrected, locals=local_dict)
    except Exception as e:
        print(f"Error in parsing the matrix element expression: {e}")
        sys.exit(1)

    # Convert the symbolic expression to a numerical function using lambdify
    try:
        matrix_element_func = sp.lambdify((mLLP, E1, E3), expr, 'numpy')
    except Exception as e:
        print(f"Error in lambdify: {e}")
        sys.exit(1)

    # Store the compiled matrix element in a list (as expected by ThreeBodyDecay.py)
    Msquared3BodyLLP = [matrix_element_func]

    # -----------------------------
    # 3. Configure Decay Modes
    # -----------------------------

    # Define PDGdecay and size_per_channel lists
    PDGdecay = [np.array(pdgs)]
    size_per_channel = [1000]  # Number of events to generate for the decay channel

    # -----------------------------
    # 4. Prepare Specific Decay Parameters
    # -----------------------------

    # Extract decay product properties
    pdg1, pdg2, pdg3 = pdgs
    # Decaying particle properties are not included in the decay_products parameters
    # So, masses, charges, stabilities correspond to the decay products
    mass1, mass2, mass3 = masses[1], masses[2], masses[3]
    charge1, charge2, charge3 = charges[1], charges[2], charges[3]
    stability1, stability2, stability3 = stabilities[1], stabilities[2], stabilities[3]

    # Create SpecificDecay tuple as expected by decay_products function
    specific_decay_params = (
        pdg1,        # PDG ID of first decay product
        pdg2,        # PDG ID of second decay product
        pdg3,        # PDG ID of third decay product
        mass1,       # Mass of first decay product
        mass2,       # Mass of second decay product
        mass3,       # Mass of third decay product
        charge1,     # Charge of first decay product
        charge2,     # Charge of second decay product
        charge3,     # Charge of third decay product
        stability1,  # Stability of first decay product
        stability2,  # Stability of second decay product
        stability3,  # Stability of third decay product
        Msquared3BodyLLP[0]  # Compiled matrix element function
    )

    # -----------------------------
    # 5. Generate Decay Events
    # -----------------------------

    # Initialize list to store all decay events
    all_decay_events = []

    # Iterate over each decay mode and generate events
    for i, decay_modes in enumerate(PDGdecay):
        pdg_list = decay_modes[decay_modes != -999]  # Filter out invalid PDGs

        if size_per_channel[i] == 0:
            continue  # Skip if no events are assigned to this decay channel

        if len(pdg_list) == 3:
            # Three-body decay
            decay_results = decay_products(
                decaying_particle_mass,      # Mass of decaying particle
                size_per_channel[i],         # Number of events to generate
                specific_decay_params        # Specific decay parameters
            )
            all_decay_events.append(decay_results)
        else:
            # This example focuses on three-body decays. Two-body and four-body decays are not handled here.
            print(f"Decay mode with PDGs {pdg_list} is not a three-body decay. Skipping.")
            continue

    # Concatenate all decay events into a single NumPy array
    if all_decay_events:
        decay_events = np.vstack(all_decay_events)
        print(f"Generated {decay_events.shape[0]} three-body decay events.")
    else:
        print("No decay events were generated.")
        sys.exit(1)

    # -----------------------------
    # 6. Extract Energy of First Decay Product
    # -----------------------------

    # The energy of the first decay product is at index 3
    E1 = decay_events[:, 3]
    print(f"Extracted energies of the first decay product (E1) for all events.")

    # -----------------------------
    # 7. Plot Histogram of E1
    # -----------------------------

    # Define histogram parameters
    num_bins = 100
    plt.figure(figsize=(10, 6))
    plt.hist(E1, bins=num_bins, color='skyblue', edgecolor='black')
    plt.title('Histogram of Energy of the First Decay Product (E1)')
    plt.xlabel('Energy E1 (GeV)')
    plt.ylabel('Number of Events')
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()

