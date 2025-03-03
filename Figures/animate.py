%%manim -qm SumOfSinusoidsVertical

import numpy as np

class SumOfSinusoidsVertical(Scene):
    def construct(self):
        # Create an axes for the composite (sum) signal at the top.
        sum_axes = Axes(
            x_range=[0, 12, 1],
            y_range=[-2, 2, 1],
            x_length=10,
            y_length=1.5,
            tips=False,
            axis_config={"color": WHITE, "include_numbers": False}
        )
        # labels = sum_axes.get_axis_labels(Tex("x-axis").scale(0.7), Text("y-axis").scale(0.45))
        sum_axes.to_edge(UP, buff=MED_SMALL_BUFF)
        axes_labels = sum_axes.get_axis_labels(MathTex(r"t").scale(0.7), Text("").scale(0.45))
        self.add(sum_axes, axes_labels)

        # Create three axes for the individual sinusoidals.
        # We use a smaller y_length so all graphs fit vertically.
        individual_config = {
            "x_range": [0, 12, 1],
            "y_range": [-1.5, 1.5, 1],
            "x_length": 10,
            "y_length": 1.3,
            "tips": False,
            "axis_config": {"color": WHITE, "include_numbers": False},
        }
        axes1 = Axes(**individual_config)
        axes2 = Axes(**individual_config)
        axes3 = Axes(**individual_config)

        # Arrange the three axes vertically.
        lower_group = VGroup(axes1, axes2, axes3)
        lower_group.arrange(DOWN, buff=0.55)
        lower_group.next_to(sum_axes, DOWN, buff=0.65)

        # Add both the top and lower axes to the scene.
        self.add(sum_axes, lower_group)

        # Define the sinusoidal functions.
        def f1(x):
            return 0.7 * np.sin(x)

        def f2(x):
            return 0.8 * np.sin(0.5 * x)

        def f3(x):
            return 0.5 * np.sin(4 * x)

        # The composite function is the pointwise sum.
        def f_sum(x):
            return f1(x) + f2(x) + f3(x) 

        # Generate x-values for smooth curves.
        x_vals = np.linspace(0, 12, 300)

        # Compute points for each graph using the appropriate axes.
        points1 = [axes1.c2p(x, f1(x)) for x in x_vals]
        points2 = [axes2.c2p(x, f2(x)) for x in x_vals]
        points3 = [axes3.c2p(x, f3(x)) for x in x_vals]
        sum_points = [sum_axes.c2p(x, f_sum(x)) for x in x_vals]

        # Create smooth VMobject graphs for each signal.
        graph1 = VMobject().set_points_smoothly(points1).set_color(RED)
        graph2 = VMobject().set_points_smoothly(points2).set_color(GREEN)
        graph3 = VMobject().set_points_smoothly(points3).set_color(YELLOW)
        sum_graph = VMobject().set_points_smoothly(sum_points).set_color(BLUE)
        
        # Animate the creation of the individual sinusoidals.
        self.play(
            Create(graph1),
            Create(graph2),
            Create(graph3),
            run_time=3
        )
        self.wait(1)

        freq_label1 = MathTex(r"\omega_1", color=RED).scale(0.6)\
            .next_to(graph1, RIGHT, buff=0.35).shift(UP * 0.1)
        freq_label2 = MathTex(r"\omega_2", color=GREEN).scale(0.6)\
            .next_to(graph2, RIGHT, buff=0.35).shift(UP * 0.1)
        freq_label3 = MathTex(r"\omega_3", color=YELLOW).scale(0.6)\
            .next_to(graph3, RIGHT, buff=0.35).shift(UP * 0.1)
        
        phase_label1_1 = MathTex(r"\theta_1^\circ", color=RED).scale(0.6)\
            .next_to(freq_label1, DOWN, buff=0.25).shift(UP * 0.1)
        phase_label2_1 = MathTex(r"\theta_2^\circ", color=GREEN).scale(0.6)\
            .next_to(freq_label2, DOWN, buff=0.25).shift(UP * 0.1)
        phase_label3_1 = MathTex(r"\theta_3^\circ", color=YELLOW).scale(0.6)\
            .next_to(freq_label3, DOWN, buff=0.25).shift(UP * 0.1)
        
        # Add labels to the scene
        self.add(freq_label1, freq_label2, freq_label3)
        self.add(phase_label1_1, phase_label2_1, phase_label3_1)
        
        label_phase_group = VGroup(freq_label1, freq_label2, freq_label3,
                                  phase_label1_1, phase_label2_1, phase_label3_1)
        # Add vertical plus signs between the individual graphs.
        # plus1 = Tex("+").scale(1)
        # plus2 = Tex("+").scale(1)
        # plus1.next_to(axes1, DOWN, buff=0.1)
        # plus2.next_to(axes2, DOWN, buff=0.1)
        # self.play(FadeIn(plus1), FadeIn(plus2))
        # self.wait(2)

        # Create the summation symbol
        sum_symbol = Tex(r"$\sum$").move_to(lower_group.get_top()).shift(UP * 0.2)
        
        # Generalized Inverse Fourier Transform representation
        sum_equation = MathTex(
            r"x(t) = \sum_{n} A_n \sin(\omega_n t + \theta_n)", color=WHITE
        ).scale(0.7).move_to(lower_group.get_top()).shift(UP * 0.2)

        # Label for the summation plot (placed at the upper-left of the sum_axes)
        xt_label = MathTex(r"x(t)", color=WHITE).scale(0.6)
        xt_label.move_to(sum_axes.get_top() + LEFT * 5.5 + UP * 0.01)  # Shift diagonally up-left
        
        # Add the label to the scene
        self.add(xt_label)
        
        # Animate the summation symbol appearing
        # self.play(Write(sum_equation))
        # self.wait(1)


        # To emphasize that the top graph is the sum, create a copy of the lower graphs,
        # then transform that copy into the composite (sum) graph.
        lower_group_copy = VGroup(graph1.copy(), graph2.copy(), graph3.copy())
        # For transformation purposes, simply ReplacementTransform into the sum_graph.
        self.play(
            ReplacementTransform(lower_group_copy, sum_graph),
            run_time=3
        )
        self.wait(3)

        # Animate the summation symbol appearing
        self.play(Write(sum_equation))
        self.wait(1)

        # Shift everything a bit to the left to prevent clipping.
        lower_axes_and_graphs = VGroup(lower_group, graph1, graph2, graph3, sum_graph, sum_equation, axes_labels, xt_label, label_phase_group)
        self.play(lower_axes_and_graphs.animate.shift(LEFT*0.7), sum_axes.animate.shift(LEFT*0.7))
        
        # ---- Add circles to show magnitude and initial phase for each pure sinusoid.
        # For f1: amplitude ~1, phase = +0.2; for f2: amplitude 0.8, phase = -0.3; for f3: amplitude 0.5, phase = +0.5.
        circle1 = Circle(radius=0.6, color=RED, stroke_width=1)
        circle2 = Circle(radius=0.7, color=GREEN, stroke_width=1)
        circle3 = Circle(radius=0.5, color=YELLOW, stroke_width=1)
        # Position the first circle to the right of axes1.
        circle1.next_to(axes1, RIGHT, buff=0.9)  # Increase spacing to avoid overlap
        circle2.next_to(axes2, RIGHT, buff=0.9)
        circle3.next_to(axes3, RIGHT, buff=0.9)
        
        # ---- Add rotating circles to show magnitude and initial phase ----
        shift_tracker = ValueTracker(0)  # Control phase rotation

        circle1 = Circle(radius=0.6, color=RED, stroke_width=1)
        circle2 = Circle(radius=0.7, color=GREEN, stroke_width=1)
        circle3 = Circle(radius=0.5, color=YELLOW, stroke_width=1)

        # Position the circles to the right of each axes.
        circle1.next_to(axes1, RIGHT, buff=1.3)
        circle2.next_to(axes2, RIGHT, buff=1.3)
        circle3.next_to(axes3, RIGHT, buff=1.3)

        # Align circles horizontally.
        x_offset = circle1.get_center()[0]
        circle2.move_to([x_offset, circle2.get_center()[1], 0])
        circle3.move_to([x_offset, circle3.get_center()[1], 0])

        # Dots for phase tracking.
        dot1 = Dot(color=RED)
        dot2 = Dot(color=GREEN)
        dot3 = Dot(color=YELLOW)
        
        dot1.move_to(circle1.point_at_angle(0))  # Starts at phase zero
        dot2.move_to(circle2.point_at_angle(0))  # Starts at phase zero
        dot3.move_to(circle3.point_at_angle(0))  # Starts at phase zero        

        # Function to rotate the dots dynamically
        def update_dot1(mob):
            mob.move_to(circle1.point_at_angle(shift_tracker.get_value()))
        def update_dot2(mob):
            mob.move_to(circle2.point_at_angle(shift_tracker.get_value() * 0.5))
        def update_dot3(mob):
            mob.move_to(circle3.point_at_angle( shift_tracker.get_value() * 4))

        dot1.add_updater(update_dot1)
        dot2.add_updater(update_dot2)
        dot3.add_updater(update_dot3)

        # # Arrows pointing from circle centers to dots
        # arrow1 = always_redraw(lambda: Arrow(circle1.get_center(), dot1.get_center(), color=RED, buff=0))
        # arrow2 = always_redraw(lambda: Arrow(circle2.get_center(), dot2.get_center(), color=GREEN, buff=0))
        # arrow3 = always_redraw(lambda: Arrow(circle3.get_center(), dot3.get_center(), color=YELLOW, buff=0))

        # Animate circles, dots, and arrows appearing
        self.play(
            FadeIn(circle1), FadeIn(dot1),
            FadeIn(circle2), FadeIn(dot2),
            FadeIn(circle3), FadeIn(dot3),
            run_time=2
        )
        self.wait(1)

        # ---- START TIME SHIFTING & ROTATING CIRCLES ----
        period = 4 * PI  # Reduce period to slow down the shift

        # Function to shift the waveforms dynamically while keeping their colors.
        def get_wave(func, axes, color):
            x_vals = np.linspace(0, 12, 300)
            pts = [axes.c2p(x, func(x - shift_tracker.get_value())) for x in x_vals]
            mob = VMobject()
            mob.set_points_smoothly(pts)
            mob.set_color(color)
            return mob

        # Apply shifting waveforms.
        wave1 = always_redraw(lambda: get_wave(f1, axes1, RED))
        wave2 = always_redraw(lambda: get_wave(f2, axes2, GREEN))
        wave3 = always_redraw(lambda: get_wave(f3, axes3, YELLOW))
        sum_wave = always_redraw(lambda: get_wave(f_sum, sum_axes, BLUE))

        # Replace static waveforms with shifting ones.
        self.remove(graph1, graph2, graph3, sum_graph)
        self.add(wave1, wave2, wave3, sum_wave)

        # ---- Display Phase Angle Inside Circles & Track Reset Count ----
        def format_phase(shift_value, frequency):
            return f"{((shift_value / (2 * PI)) * frequency * 360) % 360:.1f}^\circ"
        
        def count_cycles(shift_value, frequency):
            return f"Cycles: {int((shift_value / (2 * PI)) * frequency)}"
        
        # Cycle Counter Labels (positioned below each circle)
        cycle_count1 = always_redraw(lambda: MathTex(
            count_cycles(shift_tracker.get_value(), 1), color=RED
        ).scale(0.5).next_to(circle1, DOWN, buff=0.1))
        
        cycle_count2 = always_redraw(lambda: MathTex(
            count_cycles(shift_tracker.get_value(), 0.5), color=GREEN
        ).scale(0.5).next_to(circle2, DOWN, buff=0.1))
        
        cycle_count3 = always_redraw(lambda: MathTex(
            count_cycles(shift_tracker.get_value(), 4), color=YELLOW
        ).scale(0.5).next_to(circle3, DOWN, buff=0.1))

        # Phase Angle Labels (inside circles)
        phase_label1 = always_redraw(lambda: MathTex(
            format_phase(shift_tracker.get_value(), 1), color=RED
        ).scale(0.6).move_to(circle1.get_center()))
        
        phase_label2 = always_redraw(lambda: MathTex(
            format_phase(shift_tracker.get_value(), 0.5), color=GREEN
        ).scale(0.6).move_to(circle2.get_center()))
        
        phase_label3 = always_redraw(lambda: MathTex(
            format_phase(shift_tracker.get_value(), 4), color=YELLOW
        ).scale(0.6).move_to(circle3.get_center()))
        
        # Add phase labels & cycle counters
        self.add(phase_label1, phase_label2, phase_label3)
        self.add(cycle_count1, cycle_count2, cycle_count3)
        
        # Slider settings
        t_max = 4 * PI  # your waveform duration
        slider_x_min = sum_axes.get_right()[0] + 0.98
        slider_x_max = slider_x_min + 1.65
        slider_y = sum_axes.get_center()[1]
        
        slider_track = Line(
            np.array([slider_x_min, slider_y, 0]),
            np.array([slider_x_max, slider_y, 0]),
            color=WHITE
        )
        
        slider_dot = always_redraw(lambda: Dot(
            slider_track.point_from_proportion(shift_tracker.get_value() / (4 * PI)),
            color=BLUE
        ))
        
        # Slider label as fraction of fundamental period
        slider_label = always_redraw(lambda: MathTex(
            f"t' = {shift_tracker.get_value()/(4*PI):.2f}\\cdot T", 
            color=WHITE
        ).scale(0.6).next_to(slider_track, UP, buff=0.2))
        
        self.add(slider_track, slider_dot, slider_label)
        
        old_center_xt_label = xt_label.get_center()
        new_xt_label = MathTex(r"x(t-t^\prime)", color=WHITE).scale(0.6)
        new_xt_label.move_to(old_center_xt_label).shift(LEFT * 0.2)  # Shift slightly left

        old_center_sum = sum_equation.get_center()
        new_sum_equation = MathTex(
            r"x(t-t^\prime) =  \sum_{n} A_n \sin(\omega_n (t-t^\prime) + \theta_n)", 
            color=WHITE
        ).scale(0.65)
        # Align the new equation with the old one.
        new_sum_equation.move_to(old_center_sum)

        self.play(ReplacementTransform(xt_label, new_xt_label), ReplacementTransform(sum_equation, new_sum_equation))
        # Animate shifting and rotating circles at the same time.
        self.play(
            shift_tracker.animate.set_value(4 * PI),
            run_time=12,
            rate_func=linear
        )
        self.wait(3)

        # ---- Keep Only the Summation and Lowest-Frequency Waveform ----
        objects_to_remove = VGroup(
            # Remove waveforms for f1 and f3
            wave1, wave3,
            # Remove their axes if desired:
            axes1, axes3,
            # Remove circles and associated elements for f1 and f3:
            circle1, circle3,
            dot1, dot3,
            phase_label1, phase_label3,
            phase_label1_1, phase_label3_1,
            cycle_count1, cycle_count3,
            freq_label1, freq_label3
        )

        old_center_sum_2 = new_sum_equation.get_center()
        new_sum_equation_2 = MathTex(
            r"x(t-t^\prime) = A_2 \sin(\omega_2 (t-t^\prime) + \theta_2) + \sum_{\substack{1 \leq n < \infty, n\neq 2}} A_n \sin(\omega_n (t-t^\prime) + \theta_n)", 
            color=WHITE
        ).scale(0.65)
        # Align the new equation with the old one.
        new_sum_equation_2.move_to(old_center_sum_2)

        self.play(ReplacementTransform(new_sum_equation, new_sum_equation_2))
        
        self.play(FadeOut(objects_to_remove), run_time=2)
        self.remove(objects_to_remove)
        self.wait(1)   

        # Group these elements. (You may need to ensure that these objects were assigned to variables earlier.)
        lowest_group = VGroup(axes2, wave2, circle2, dot2, phase_label2, phase_label2_1, cycle_count2, freq_label2)
        
        # Animate shifting the entire lowest-frequency group upward (e.g., by 1 unit)
        self.play(lowest_group.animate.shift(UP * 1.2), run_time=1)
        self.wait(1)

        # old_center = sum_equation.get_center()
        # new_sum_equation = MathTex(
        #     r"x(t) = A_2 \sin(\omega_2 t + \theta_2) + \sum_{\substack{1 \leq n \leq \infty, n\neq 2}} A_n \sin(\omega_n t + \theta_n)", 
        #     color=WHITE
        # ).scale(0.7)
        # # Align the new equation with the old one.
        # new_sum_equation.move_to(old_center)
        
        # Now, change the colors of the appropriate parts.
        new_sum_equation_2[0][14:16].set_color(GREEN)  # for \omega_2
        new_sum_equation_2[0][8:11].set_color(GREEN)  # for \theta_2
        new_sum_equation_2[0][23:25].set_color(GREEN)  # for \theta_2
        # Transform the old sum_equation into the new one
        self.wait(1)

        # Now, add your desired text below this group.
        text_below = Tex(
            r"Shift operations form an Abelian group for each harmonic, allowing us to uniquely represent each point in the shift space using the phase angle of a harmonic whose period is equal to or longer than the length of sample. This ensures that samples, under different shifts, can be mapped to unique points in the data manifold, preserving task-related information.",
            color=WHITE
        ).scale(0.8).to_edge(DOWN)
        text_below.scale(0.8)
        # Position the text below the moved lowest_group
        text_below.next_to(lowest_group, DOWN, buff=0.45)
        self.play(FadeIn(text_below))
        self.wait(3)

        # Animate the shift in discrete steps so that the phase (and waveform) update uniquely
        self.play(shift_tracker.animate.set_value(1.20 * PI), run_time=3, rate_func=linear)
        self.wait(2)  # Pause to clearly show the phase value and waveform update
        
        self.play(shift_tracker.animate.set_value(2.30 * PI), run_time=3, rate_func=linear)
        self.wait(2)
        
        self.play(shift_tracker.animate.set_value(4 * PI), run_time=3, rate_func=linear)
        self.wait(3)

######################

%%manim -qm MLPWithWaveformInput

import numpy as np
from manim import *

class MLPWithWaveformInput(Scene):
    def create_nodes(self, x_shift, y_shift, num_nodes, node_fill_color=GRAY, layer_output=None):
        node_group = VGroup()
        for i in range(num_nodes):
            # Fill nodes with a constant opacity.
            node = Circle(
                radius=0.35,
                stroke_color=WHITE,
                stroke_width=3,
                fill_color=node_fill_color,
                fill_opacity=0.6
            )
            node_group.add(node)
        node_group.arrange(DOWN, buff=0.15)
        node_group.shift(x_shift * RIGHT + y_shift * DOWN)
        return node_group

    def get_connection_line(self, node1, node2):
        # Connect nodes with a line (not an arrow)
        start = node1.get_center()
        end = node2.get_center()
        direction = (end - start) / np.linalg.norm(end - start)
        # Adjust the endpoints so they lie on the boundaries
        r1 = node1.get_width() / 1.5
        r2 = node2.get_width() / 1.5
        start_point = start + direction * r1
        end_point = end - direction * r2
        line = Line(start_point, end_point, color=GRAY)
        line.set_stroke(width=2.3)
        return line

    def construct(self):
        # --- Define the sinusoidal functions ---
        def f1(x):
            return 0.4 * np.sin(x)
        def f2(x):
            return -0.3 * np.sin(0.5 * x)
        def f3(x):
            return 0.5 * np.sin(4 * x)
        # The composite function is the pointwise sum.
        def f_sum(x):
            return f1(x) + f2(x) + f3(x)

        # Define a global shift tracker.
        shift_tracker = ValueTracker(0)

        # -------------------------------
        # 2. Create the MLP layers.
        # Input: 3 nodes, Hidden: 2 nodes, Output: 1 node.
        input_layer = self.create_nodes(-1.5, 0, 3, node_fill_color=BLUE_B)
        hidden_layer = self.create_nodes(0, 0, 2, node_fill_color=BLUE_B)
        output_layer = self.create_nodes(1.5, 0, 1, node_fill_color=BLUE_B)

        # # Animate creation.
        # self.play(Create(input_layer), Create(hidden_layer), Create(output_layer))
        # self.wait(1)

        # Connect layers with lines.
        connections = VGroup()
        for n1 in input_layer:
            for n2 in hidden_layer:
                connections.add(self.get_connection_line(n1, n2))
        for n1 in hidden_layer:
            for n2 in output_layer:
                connections.add(self.get_connection_line(n1, n2))
                
        self.play(Create(input_layer), Create(hidden_layer), Create(output_layer), Create(connections))

        # Group the NN and shift it right (compact layout).
        nn_group = VGroup(input_layer, hidden_layer, output_layer, connections)
        self.play(nn_group.animate.shift(RIGHT * 3), run_time=0.5)
        self.wait(1)

        # Draw a surrounding rectangle around the NN.
        nn_rect = SurroundingRectangle(nn_group, buff=0.25, color=WHITE)
        self.play(Create(nn_rect))
        self.wait(0.5)
        # Add a label "f_{\theta}" above the NN rectangle.
        f_label = MathTex(r"f_{\theta}", color=WHITE).scale(1)
        f_label.next_to(nn_rect, UP, buff=0.2)
        self.play(Write(f_label))
        self.wait(1)

        # -------------------------------
        # 3. Create the input waveform.
        input_axes = Axes(
            x_range=[0, 12, 1],
            y_range=[-1.5, 1.5, 0.5],
            x_length=6,
            y_length=3,
            tips=False,
            axis_config={"color": WHITE}
        ).to_edge(LEFT, buff=1)
        self.add(input_axes)
        
        axes_labels = input_axes.get_axis_labels(MathTex(r"t").scale(0.7), MathTex(r"x(t)").scale(0.7))
        self.add(input_axes, axes_labels)
        
        # Label the input waveform "X".
        x_label = MathTex("X", color=WHITE).scale(1)
        x_label.next_to(input_axes, UP, buff=0.2)
        self.play(Write(x_label))

        # --- Change the waveform to the composite sum ---
        # Instead of using np.sin(t - shift), we now use f_sum(t - shift).
        def input_wave(t, shift):
            return f_sum(t - shift)
            
        x_vals = np.linspace(0, 12, 300)
        waveform = always_redraw(lambda: VMobject().set_points_smoothly(
            [input_axes.c2p(x, input_wave(x, shift_tracker.get_value())) for x in x_vals]
        ).set_color(YELLOW))
        self.add(waveform)
        self.wait(1)

        # Shift the entire NN (and its rectangle and f_label) upward to create room for text below.
        self.play(
            AnimationGroup(
                nn_group.animate.shift(UP * 1),
                nn_rect.animate.shift(UP * 1),
                f_label.animate.shift(UP * 1),
                axes_labels.animate.shift(UP * 1),
                input_axes.animate.shift(UP * 1),
                x_label.animate.shift(UP * 1)
            ),
            run_time=0.5
        )
        self.wait(0.5)

        # -------------------------------
        # 4. Show the output prediction.
        # Instead of a numeric output, we use class labels.
        def output_activation(shift):
            return np.cos(shift)
        def output_class(shift):
            val = output_activation(shift)
            # Partition the range [-1,1] into 4 regions.
            if val >= 0.75:
                return "C1", RED
            elif val >= 0.25:
                return "C2", ORANGE
            elif val >= -0.25:
                return "C3", YELLOW
            else:
                return "C4", BLUE

        # Create four class labels and arrange them vertically.
        label1 = MathTex("C1", color=RED).scale(0.8)
        label2 = MathTex("C2", color=ORANGE).scale(0.8)
        label3 = MathTex("C3", color=YELLOW).scale(0.8)
        label4 = MathTex("C4", color=BLUE).scale(0.8)
        class_labels = VGroup(label1, label2, label3, label4)
        class_labels.arrange(DOWN, buff=0.3)
        # Position the labels centered vertically relative to the NN rectangle.
        class_labels.next_to(nn_rect, RIGHT, buff=0.5)
        
        # Updater to set opacity: the active class is fully opaque; others are faded.
        def update_class_labels(mob):
            active_class, _ = output_class(shift_tracker.get_value())
            for label in mob:
                if label.get_tex_string() == active_class:
                    label.set_opacity(1)
                else:
                    label.set_opacity(0.3)
            mob.arrange(DOWN, buff=0.3)
            mob.next_to(nn_rect, RIGHT, buff=0.5)
        class_labels.add_updater(update_class_labels)
        self.add(class_labels)
        
        # --- Place a "Y" label above the class labels.
        y_label = MathTex("Y", color=WHITE).scale(1)
        y_label.next_to(class_labels, UP, buff=0.7)
        self.play(Write(y_label))
        self.wait(1)
        
        # -------------------------------
        # 5. Animate the shift in discrete steps.
        # Start the explanatory text at the same time as the shift.
        explanation = Tex(
            r"Neural networks, $ f_{\theta} : X \rightarrow Y $, change their predictions when the input signal is shifted, even though the attributed features (magnitude, frequency) remain unchanged. To overcome this, we introduced a diffeomorphism.",
            font_size=36,
            color=WHITE
        )

        explanation.arrange(DOWN, aligned_edge=LEFT)
        # Position explanation at the bottom of the screen.
        explanation.to_edge(DOWN, buff=0.85)
        
        # Animate the first discrete shift simultaneously with writing the explanation.
        self.play(
            AnimationGroup(
                shift_tracker.animate.set_value(PI),
                lag_ratio=0.2
            ),
            run_time=3, rate_func=linear
        )
        self.wait(1)
        self.play(shift_tracker.animate.set_value(2*PI), run_time=3, rate_func=linear)
        self.wait(1)
        self.play(FadeIn(explanation),run_time=3,rate_func=linear)
        self.play(shift_tracker.animate.set_value(3*PI), run_time=3, rate_func=linear)
        self.wait(1)
        self.play(shift_tracker.animate.set_value(4*PI), run_time=3, rate_func=linear)
        self.wait(3)
        






