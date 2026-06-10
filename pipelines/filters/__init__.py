"""Pipe-and-Filter data preparation stages.

Each filter: pd.DataFrame (validated schema) -> pd.DataFrame (output schema)

Order: sampler -> validator -> labeller -> classifier -> encoder -> splitter
"""
