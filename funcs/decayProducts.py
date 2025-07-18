# decayProducts.py

import sys
import os
import numpy as np
import time  # Import time module for timing

#Path to the library containing pythia8.so
sys.path.insert(0, '/home/name/Downloads/pythia8312/lib')

# Import Pythia8 and other required modules
import pythia8

from . import TwoBodyDecay, ThreeBodyDecay, FourBodyDecay
from . import PDG
from . import inelasticDMCalcs

def distribute_events(total_events, branching_ratios):
    """
    Distributes the total number of events among different decay channels according to their branching ratios.
    """
    branching_ratios = np.array(branching_ratios)
    branching_ratios /= np.sum(branching_ratios)  # Normalize branching ratios
    events = np.array([total_events * ratio for ratio in branching_ratios])
    rounded_events = np.round(events).astype(int)
    rounded_events = np.where(branching_ratios == 0, 0, rounded_events)
    difference = total_events - np.sum(rounded_events)
    
    # Adjust the difference to ensure total events match
    while difference != 0:
        for i in range(len(branching_ratios)):
            if difference == 0:
                break
            if branching_ratios[i] > 0:
                if difference > 0 and (rounded_events[i] < events[i] or np.sum(rounded_events) < total_events):
                    rounded_events[i] += 1
                    difference -= 1
                elif difference < 0 and rounded_events[i] > 0:
                    rounded_events[i] -= 1
                    difference += 1
    return rounded_events

def simulateDecays_rest_frame(mass, PDGdecay, BrRatio, size, Msquared3BodyLLP, selected_decay_indices, br_visible_val, process_in_pythia=True):
    """
    Simulates the decays of a particle in its rest frame, distributing events among selected decay channels.
    """
    def get_particle_properties(pdg_list):
        masses = [PDG.get_mass(pdg) for pdg in pdg_list]
        charges = [PDG.get_charge(pdg) for pdg in pdg_list]
        stabilities = [PDG.get_stability(pdg) for pdg in pdg_list]
        return masses, charges, stabilities

    # Start total timing
    total_start_time = time.time()

    # Get branching ratios and decay modes for selected channels
    selected_BrRatios = [BrRatio[idx] for idx in selected_decay_indices]
    selected_PDGdecay = [PDGdecay[idx] for idx in selected_decay_indices]
    if Msquared3BodyLLP is not None:
        selected_Msquared3BodyLLP = [Msquared3BodyLLP[idx] for idx in selected_decay_indices]
    else:
        selected_Msquared3BodyLLP = [None] * len(selected_BrRatios)

    # Normalize the branching ratios
    normalized_br_ratios = [br / br_visible_val for br in selected_BrRatios]

    # Distribute events among the selected decay channels
    size_per_channel = distribute_events(size, normalized_br_ratios)
    print(f"Events per selected decay channel: {size_per_channel}")
    all_decay_events = []

    # Start timing for decay simulations
    decay_sim_start = time.time()

    # Loop through selected decay channels
    for idx_in_selected, (decay_modes, channel_size) in enumerate(zip(selected_PDGdecay, size_per_channel)):
        if channel_size == 0:
            continue  # Skip if no events are assigned to this decay channel

        pdg_list = decay_modes[decay_modes != -999]
        i = selected_decay_indices[idx_in_selected]

        if len(pdg_list) == 2:
            # Two-body decay
            pdg1, pdg2 = pdg_list
            masses, charges, stabilities = get_particle_properties([pdg1, pdg2])
            decay_results = TwoBodyDecay.decay_products(
                mass, channel_size, masses[0], masses[1], pdg1, pdg2,
                charges[0], charges[1], stabilities[0], stabilities[1]
            )
        elif len(pdg_list) == 3:
            # Three-body decay
            pdg1, pdg2, pdg3 = pdg_list
            masses, charges, stabilities = get_particle_properties([pdg1, pdg2, pdg3])
            specific_decay_params = (
                pdg1, pdg2, pdg3, masses[0], masses[1], masses[2],
                charges[0], charges[1], charges[2],
                stabilities[0], stabilities[1], stabilities[2],
                selected_Msquared3BodyLLP[idx_in_selected]
            )
            decay_results = ThreeBodyDecay.decay_products(
                mass, channel_size, specific_decay_params
            )
        elif len(pdg_list) == 4:
            # Four-body decay
            pdg1, pdg2, pdg3, pdg4 = pdg_list
            masses, charges, stabilities = get_particle_properties([pdg1, pdg2, pdg3, pdg4])
            specific_decay_params = (
                pdg1, pdg2, pdg3, pdg4, masses[0], masses[1], masses[2], masses[3],
                charges[0], charges[1], charges[2], charges[3],
                stabilities[0], stabilities[1], stabilities[2], stabilities[3]
            )
            decay_results = FourBodyDecay.decay_products(
                mass, specific_decay_params, channel_size
            )
        else:
            print(f"Invalid number of decay products ({len(pdg_list)}) in decay channel {i}. Skipping.")
            continue  # Skip invalid decay channels

        # Append each decay event individually
        for decay_event in decay_results:
            all_decay_events.append(decay_event)

    # End timing for decay simulations
    decay_sim_end = time.time()
    decay_sim_time = decay_sim_end - decay_sim_start
    print(f"\nTotal decay events generated: {len(all_decay_events)}")
    print(f"Time taken for decay simulations: {decay_sim_time:.2f} seconds")

    if process_in_pythia:
        # Start timing for Pythia processing
        pythia_start_time = time.time()

        # Process all decay events through Pythia sequentially
        if all_decay_events:
            print("\nStarting Pythia processing of decay events...")
            processed_results = process_events_with_pythia(all_decay_events, mass)
        else:
            print("\nNo decay events to process with Pythia.")
            processed_results = []

        # End timing for Pythia processing
        pythia_end_time = time.time()
        pythia_time = pythia_end_time - pythia_start_time
        print(f"Time taken for Pythia processing: {pythia_time:.2f} seconds")

    else:
        print("\nNo decay events to process with Pythia.")
        processed_results = _strip_and_pad(all_decay_events)

    # Total time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total time for simulateDecays_rest_frame: {total_time:.2f} seconds")

    return (processed_results, size_per_channel)

def process_events_with_pythia(decay_events_list, mass):
    """
    Processes a list of decay events through Pythia sequentially, pads each event to have the same number of particles,
    and writes the processed events to external files.
    """
    # Initialize Pythia
    pythia = pythia8.Pythia()

    # Register chi and chiPr particles in pythia
    inelasticDMCalcs.register_stable_particle(pythia, inelasticDMCalcs.CHI_PDG, inelasticDMCalcs.CHI_NAME)
    inelasticDMCalcs.register_stable_particle(pythia, inelasticDMCalcs.CHI_PR_PDG, inelasticDMCalcs.CHI_PR_PDG)

    #pythia.readString("Print:quiet = on")  # Suppress banners and output
    pythia.readString("ProcessLevel:all = off")
    pythia.readString("PartonLevel:all = on")
    pythia.readString("HadronLevel:all = on")
    # Keep certain particles stable
    stable_particles = [13, -13, 211, -211, 321, -321, 130, inelasticDMCalcs.CHI_PR_PDG, inelasticDMCalcs.CHI_PDG]
    for pid in stable_particles:
        pythia.readString(f"{pid}:mayDecay = off")
    mother_id = 25  # PDG ID for the mother particle (e.g., Higgs boson)
    pythia.init()

    processed_events = []
    event_counter = 0  # Initialize the event counter

    for decay in decay_events_list:
        event_counter += 1

        pythia.event.reset()
        num_particles = len(decay) // 8  # Each particle has 8 attributes

        # Set mother particle at rest
        mother_px, mother_py, mother_pz = 0.0, 0.0, 0.0
        mother_e = mass
        mother_m = mass

        # Append the mother particle with status -23
        pythia.event.append(
            mother_id, -23, 0, 0, 2, num_particles + 1,
            0, 0,
            mother_px, mother_py, mother_pz, mother_e, mother_m
        )

        # Process decay products
        for i_particle in range(num_particles):
            idx = i_particle * 8
            px = decay[idx]
            py = decay[idx + 1]
            pz = decay[idx + 2]
            e = decay[idx + 3]
            m = decay[idx + 4]
            pdgId = int(decay[idx + 5])

            if pdgId == -999:
                continue  # Skip placeholder entries

            # Default color tags
            col = 0
            acol = 0

            # Assign colors based on PDG ID
            if pdgId in [1, 2, 3, 4, 5]:
                # Quarks
                col = 501
                acol = 0
                status_code = 23  # Outgoing parton to be showered
            elif pdgId in [-1, -2, -3, -4, -5]:
                # Antiquarks
                col = 0
                acol = 501
                status_code = 23  # Outgoing parton to be showered
            elif pdgId == 21:
                # Gluon
                # Ensure that gluons come in pairs
                if i_particle % 2 == 0:
                    col = 501
                    acol = 502
                else:
                    # For the second gluon
                    col = 502
                    acol = 501
                status_code = 23  # Outgoing parton to be showered
            else:
                # Other particles
                col = 0
                acol = 0
                status_code = 1  # Final-state particle

            # Append particle to Pythia event
            # Set mother indices to the mother particle (index 1)
            pythia.event.append(
                pdgId, status_code, 1, 1, 0, 0,
                col, acol,
                px, py, pz, e, m
            )

        # Process the event through Pythia
        if not pythia.forceHadronLevel():
            #uncomment the line below is you want to see the decay products chain resulting by processing the decay in pythia during the simulation
            #pythia.event.list()
            pass

        # Extract final state particles
        final_state_particles = []
        for p in pythia.event:
            if p.isFinal():
                particle_data = [
                    p.px(), p.py(), p.pz(), p.e(), p.m(), p.id()
                ]
                final_state_particles.extend(particle_data)

        processed_events.append(final_state_particles)

    if not processed_events:
        print("\nNo processed events were generated.")

    # After processing all events, determine the maximum number of particles in any event
    max_n = max((len(event) // 6 for event in processed_events), default=0)

    # Define the padding to append for each missing particle
    padding = [0.0, 0.0, 0.0, 0.0, 0.0, -999]

    # Pad each event to have the same number of particles
    padded_events = []
    for event in processed_events:
        n = len(event) // 6
        m = max_n - n
        if m > 0:
            event_padded = event + padding * m
        else:
            event_padded = event
        padded_events.append(event_padded)

    # Convert the list of padded events to a NumPy array for efficient storage and computation
    padded_events_array = np.array(padded_events)

    return padded_events_array

def _strip_and_pad(decay_events_list):
    """
    Convert internal (8-slot) decay records to the 6-slot records normally
    returned after Pythia, and pad every event to the same particle count.
    """
    processed = []
    for ev in decay_events_list:
        out = []
        n = len(ev) // 8
        for i in range(n):
            base = 8 * i
            # copy px,py,pz,E,m  (slots 0..4) + pdgId (slot 5)
            out.extend(ev[base : base + 5])
            out.append(ev[base + 5])
        processed.append(out)

    # ------------------------------------------------------------------ pad --
    max_n = max(len(ev) // 6 for ev in processed) if processed else 0
    pad_tpl = [0.0, 0.0, 0.0, 0.0, 0.0, -999]               # one "blank" part.
    padded = []
    for ev in processed:
        miss = max_n - len(ev) // 6
        if miss:
            ev = list(ev) + pad_tpl * miss
        padded.append(ev)

    return np.asarray(padded, dtype=float)