class NotApplicable(Exception):
    """The request is outside what turbo-socialsci implements (an option, data layout or estimator it
    does not cover). Nothing was computed; do the task with another tool."""
