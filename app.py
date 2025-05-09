import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

import core_algorithm as ca


def main() -> None:
    st.set_page_config(
        page_title="Genetic Algorithm App",
        page_icon=":guardsman:",
    )
    st.title(f"Genetic Algorithm on a 2D Function")

    # sidebar - GA parameters
    with st.sidebar:
        st.header("Select Parameters")
        selected_func_type = st.selectbox("Target Function", ca.FunctionBox.values())
        pop_size = st.number_input("Population Size", value=100, min_value=10, max_value=2000, step=10)
        epochs = st.number_input("Number of Epochs", value=20, min_value=1, max_value=500, step=1)
        selection_method = st.selectbox("Selection Method", ca.SelectionBox.values())
        crossover_method = st.selectbox("Crossover Method", ca.CrossMethodBox.values())
        crossover_rate = st.slider("Crossover Rate", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
        alpha_cross_var = st.slider("Alpha value for crossover method", min_value=0.0, max_value=1.0, value=0.25, step=0.05, disabled=not (crossover_method == ca.CrossMethodBox.TYPEALPHAMIX or crossover_method == ca.CrossMethodBox.TYPEALPHABETAMIX))
        beta_cross_var = st.slider("Beta value for crossover method", min_value=0.0, max_value=1.0, value=0.25, step=0.05, disabled=(crossover_method != ca.CrossMethodBox.TYPEALPHABETAMIX))
        mutation_method = st.selectbox("Mutation Method", ca.MutationMethodBox.values())
        mutation_rate = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        min_mutation = st.number_input("Min value for isosceles mutation", value=-10.0, disabled=(mutation_method != ca.MutationMethodBox.ISOSCELES))
        max_mutation = st.number_input("Max value for isosceles mutation", value=10.0, disabled=(mutation_method != ca.MutationMethodBox.ISOSCELES))
        loc = st.number_input("Mean value for gausian mutation", value=10, disabled=(mutation_method != ca.MutationMethodBox.GAUSSIAN))
        scale = st.number_input("Standard deviation for gausian mutation", value=1, disabled=(mutation_method != ca.MutationMethodBox.GAUSSIAN))

        # button to run the GA
        run_button = st.button("Run", disabled=((min_mutation >= max_mutation) and (mutation_method == ca.MutationMethodBox.ISOSCELES)) or ((scale < 0) and (mutation_method == ca.MutationMethodBox.GAUSSIAN)))

        if min_mutation >= max_mutation and mutation_method == ca.MutationMethodBox.ISOSCELES:
            st.warning("Min value should be less than max value. Please correct the values.")

        if scale < 0 and mutation_method == ca.MutationMethodBox.GAUSSIAN:
            st.warning("Standard deviation can't be less than 0")
    
    
    match selected_func_type:
        case ca.FunctionBox.RASTRIGIN:
            func = ca.RastriginFunction()
        case ca.FunctionBox.HYPERSPHERE:
            func = ca.HypersphereFunction()
        case ca.FunctionBox.ROSENBROCK:
            func = ca.RosenbrockFunction()
        case ca.FunctionBox.Styblinski_and_Tang:
            func = ca.Styblinski_and_Tang()
        case _:
            raise ValueError(f"Selected invalid function type: {selected_func_type}")

    if run_button:
        # progress bar
        progress_bar = st.progress(0)

        best_individuals_per_epoch = []
        best_scores_per_epoch = []

        # run GA with a manual loop so we can update progress
        population = ca.initialize_population(pop_size)
        for epoch in range(epochs):
            # evaluate fitness for each individual
            scores = [ca.fitness(func, ind) for ind in population]
            best_score_epoch = max(scores)
            best_ind_epoch = population[np.argmax(scores)]

            best_individuals_per_epoch.append(best_ind_epoch)
            best_scores_per_epoch.append(best_score_epoch)

            # create new population
            new_population = []
            while len(new_population) < pop_size:
                parent1 = ca.selection(population, scores, method=selection_method)
                parent2 = ca.selection(population, scores, method=selection_method)
                cross_result = ca.crossover(parent1, parent2, crossover_rate, crossover_method, alpha_cross_var, beta_cross_var, func)
                if crossover_method == ca.CrossMethodBox.GRAIN or crossover_method == ca.CrossMethodBox.AVERAGE:
                    child1 = cross_result
                    child1 = ca.mutate(child1, mutation_method, mutation_rate, min_mutation, max_mutation, loc, scale)
                    new_population.extend([child1])
                else:
                    child1, child2 = cross_result
                    child1 = ca.mutate(child1, mutation_method, mutation_rate, min_mutation, max_mutation, loc, scale)
                    child2 = ca.mutate(child2, mutation_method, mutation_rate, min_mutation, max_mutation, loc, scale)
                    new_population.extend([child1, child2])

            population = new_population[:pop_size]
            
            progress_bar.progress(int((epoch + 1) / epochs * 100))
        
        # # remove progress bar after GA completes
        progress_bar.empty()  

        # Compute overall best
        best_overall_ind = best_individuals_per_epoch[np.argmax(best_scores_per_epoch)]

        # generate 3D Plot
        x_vals = np.linspace(*ca.Bounds, num=500)
        y_vals = np.linspace(*ca.Bounds, num=500)
        x, y = np.meshgrid(x_vals, y_vals)
        z = func(x, y)

        fig = go.Figure(
            data=[
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    colorscale='Viridis',
                    opacity=0.7,
                    name='Func Surface'
                ),
            ],
        )

        fig.add_trace(go.Scatter3d(
            x=[ind[0] for ind in best_individuals_per_epoch],
            y=[ind[1] for ind in best_individuals_per_epoch],
            z=[func(x, y) for x, y in best_individuals_per_epoch],
            mode='markers+lines',
            marker=dict(
                size=5,
                color='red',
                symbol='circle'
            ),
            name='Best per epoch'
        ))

        fig.add_trace(go.Scatter3d(
            x=[func.global_minimum()[0]],
            y=[func.global_minimum()[1]],
            z=[func(*func.global_minimum())],
            mode='markers',
            marker=dict(
                size=10,
                color='blue',
                symbol='circle'
            ),
            name='Global minimum',
        ))

        fig.add_trace(go.Scatter3d(
            x=[best_overall_ind[0]],
            y=[best_overall_ind[1]],
            z=[func(*best_overall_ind)],
            mode='markers',
            marker=dict(
                size=8,
                color='green',
                symbol='circle'
            ),
            name='Found minimum',
        ))

        fig.update_layout(
            title=f"Genetic Algorithm path on {selected_func_type} function",
            scene=dict(
                xaxis=dict(
                    range=[ca.Bounds[0], ca.Bounds[1]],  # Zakres dla osi X
                    title="X",
                ),
                yaxis=dict(
                    range=[ca.Bounds[0], ca.Bounds[1]],  # Zakres dla osi Y
                    title="Y",
                ),
                zaxis=dict(
                    range=[round(z.min()) - 1, round(z.max()) + 1],  # Zakres dla osi Z
                    title="f(X,Y)",
                ),
            ),
            width=700,
            height=700,
            legend=dict(
                orientation="h",  # horizontal legend
                yanchor="bottom",
                y=1.02,  # Position above the plot
                xanchor="center",
                x=0.5,  # Centered horizontally
                bgcolor="rgba(255,255,255,0.8)"  # Semi-transparent background
            )
        )

        # Store results in session state
        st.session_state.report_df = pd.DataFrame(
            {
                "Epoch": range(1, epochs + 1),
                "Best Individual (X)": [ind[0] for ind in best_individuals_per_epoch],
                "Best Individual (Y)": [ind[1] for ind in best_individuals_per_epoch],
                "Best Score": best_scores_per_epoch,
            }
        )

        st.write("Best Overall Individual:")
        st.header(f"f(x, y) = ({best_overall_ind[0]:.3f}, {best_overall_ind[1]:.3f}) = {func(*best_overall_ind):.6f}")
        
        st.plotly_chart(fig, use_container_width=True)


    if 'report_df' in st.session_state:
        if not os.path.exists("reports"):
            os.mkdir("reports")

        generate_report = st.button("Generate Report")
        if generate_report:
            st.session_state.report_df.to_csv(
                f"reports/func{selected_func_type}_epoch{epochs}_population{pop_size}.csv", 
                index=False
            )
            st.success("Report generated successfully!")
    

if __name__ == "__main__":
    main()
