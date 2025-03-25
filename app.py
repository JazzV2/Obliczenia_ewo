import numpy as np
import plotly.graph_objects as go
import streamlit as st

import core_algorithm as ca


def main() -> None:
    st.title("Genetic Algorithm Demo on a 2D Function")

    # Sidebar - GA parameters
    with st.sidebar:
        st.header("Genetic Algorithm Parameters")
        selected_func_type = st.selectbox("Target Function", ca.FunctionBox.values())
        pop_size = st.number_input("Population Size", value=50, min_value=2, max_value=1000)
        epochs = st.number_input("Number of Epochs", value=20, min_value=1, max_value=1000)
        selection_method = st.selectbox("Selection Method", ca.SelectionBox.values())
        crossover_rate = st.slider("Crossover Rate", min_value=0.0, max_value=1.0, value=0.9)
        mutation_rate = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.1)

        # Button to run the GA
        run_button = st.button("Run")

    st.markdown(f"{selected_func_type} function")
    match selected_func_type:
        case ca.FunctionBox.RASTRIGIN.value:
            func = ca.RastriginFunction()
        case ca.FunctionBox.HYPERSPHERE.value:
            func = ca.HypersphereFunction()
        case ca.FunctionBox.ROSENBROCK.value:
            func = ca.RosenbrockFunction()
        case _:
            raise ValueError(f"Selected invalid function type: {selected_func_type}")

    if run_button:
        # Progress bar
        progress_bar = st.progress(0)

        best_individuals_per_epoch = []
        best_scores_per_epoch = []

        # Run the GA with a manual loop so we can update progress
        population = ca.initialize_population(pop_size)
        for epoch in range(epochs):
            # Evaluate fitness for each individual
            scores = [ca.fitness(func, ind) for ind in population]
            best_score_epoch = max(scores)
            best_ind_epoch = population[np.argmax(scores)]

            best_individuals_per_epoch.append(best_ind_epoch)
            best_scores_per_epoch.append(best_score_epoch)

            # Create new population
            new_population = []
            while len(new_population) < pop_size:
                parent1 = ca.selection(population, scores, method=selection_method)
                parent2 = ca.selection(population, scores, method=selection_method)
                child1, child2 = ca.crossover(parent1, parent2, crossover_rate)
                child1 = ca.mutate(child1, mutation_rate)
                child2 = ca.mutate(child2, mutation_rate)
                new_population.extend([child1, child2])

            population = new_population[:pop_size]
            
            # Update progress bar
            progress_bar.progress(int((epoch + 1) / epochs * 100))
        
        # After GA completes
        progress_bar.empty()  # remove progress bar

        # Compute overall best
        best_overall_ind = best_individuals_per_epoch[np.argmax(best_scores_per_epoch)]

        # Prepare data for 3D Plot
        # We'll generate a grid of x,y points and compute Rastrigin for the surface
        x_vals = np.linspace(*ca.Bounds, num=500)
        y_vals = np.linspace(*ca.Bounds, num=500)
        x, y = np.meshgrid(x_vals, y_vals)
        z = func(x, y)

        # Plotly figure
        fig = go.Figure(
            data=[
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    colorscale='Viridis',
                    opacity=0.7,
                    name='Rastrigin Surface'
                ),
            ]
        )

        # Add scatter points for best individuals each epoch
        xs = [ind[0] for ind in best_individuals_per_epoch]
        ys = [ind[1] for ind in best_individuals_per_epoch]
        zs = [func(x, y) for x, y in best_individuals_per_epoch]

        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers+lines',
            marker=dict(
                size=5,
                color='red',
                symbol='circle'
            ),
            name='Best individual per epoch'
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
            name='Global minimum'
        ))

        fig.update_layout(
            title="Rastrigin Function Optimization with GA",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='f(X,Y)'
            ),
            width=700,
            height=700
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Display best overall individual
        st.subheader("Best Overall Individual")
        st.write(f"Coordinates (x, y): {best_overall_ind}")
        st.write(f"Rastrigin Value: {func(*best_overall_ind):.4f}")


if __name__ == "__main__":
    main()
