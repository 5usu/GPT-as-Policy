"""KUKA LBR iisy 11 R1300 extension to the RoboDojo hybrid-rollout flow.

Adapts the upstream observation -> proposal -> FK preview -> gate review ->
student/edit/eef decision loop to a single 6-DoF KUKA arm driven over RSI.

REAL MOTION IS IMPOSSIBLE BY DEFAULT. Every command path is closed unless a
chain of explicit gates is satisfied (see safety.py), and the default build
emits nothing to any robot.
"""
