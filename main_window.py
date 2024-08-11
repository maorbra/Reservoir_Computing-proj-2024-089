import os
import logging
from pymeasure.display.windows import ManagedWindowBase
from pymeasure.display.widgets import PlotWidget, TableWidget, LogWidget
from pymeasure.experiment import Results, unique_filename

from config import RESULTS_DIR

log = logging.getLogger('')


class MainWindow(ManagedWindowBase):

    def __init__(
            self,
            procedure_class,
            inputs=(),
            displays=(),
            sequencer_inputs=None,
            filename_prefix="DATA"
    ):
        self.procedure_class = procedure_class
        self.filename_prefix = filename_prefix
        widget_list = (
            PlotWidget(
                "Experiment Plot",
                self.procedure_class.DATA_COLUMNS,
            ),
            TableWidget("Experiment Table", self.procedure_class.DATA_COLUMNS),
            LogWidget("Experiment Log"),
        )
        super().__init__(
            procedure_class=self.procedure_class,
            inputs=inputs,
            displays=displays,
            widget_list=widget_list,
            sequencer=bool(sequencer_inputs),
            sequencer_inputs=sequencer_inputs,
        )
        logging.getLogger().addHandler(widget_list[-1].handler)
        log.setLevel(self.log_level)
        log.info("ManagedWindow connected to logging")
        window_title = self.procedure_class.__name__
        self.setWindowTitle(window_title)

    def queue(self, procedure=None):
        if not os.path.exists(RESULTS_DIR):
            os.mkdir(RESULTS_DIR)
        filename = unique_filename(RESULTS_DIR, self.filename_prefix)

        if procedure is None:
            procedure = self.make_procedure()

        results = Results(procedure, filename)
        experiment = self.new_experiment(results)

        self.manager.queue(experiment)
