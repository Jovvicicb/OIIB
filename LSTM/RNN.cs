using CNTK;
using LSTM.Model;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LSTM
{
    public class RNN
    {
        public static Function RecurrenceLSTM(Variable input, int outputDim,
             int cellDim, DataType dataType, DeviceDescriptor device, bool returnSequence = false,
            Activation actFun = Activation.Tanh, bool usePeephole = true,
                                bool useStabilizer = true, uint seed = 1)
        {
            if (outputDim <= 0 || cellDim <= 0)
        throw new Exception("Dimension of LSTM cell cannot be zero.");
            //prepare output and cell dimensions 
            NDShape hShape = new int[] { outputDim };
            NDShape cShape = new int[] { cellDim };

            //create placeholders
            //Define previous output and previous cell state as placeholder 
            //which will be replaced with past values later
            var dh = Variable.PlaceholderVariable(hShape, input.DynamicAxes);
            var dc = Variable.PlaceholderVariable(cShape, input.DynamicAxes);

            //create lstm cell
            var lstmCell = new LSTM(input, dh, dc, dataType, actFun,
                                    usePeephole, useStabilizer, seed, device);

            //get actual values of output and cell state
            var actualDh = CNTKLib.PastValue(lstmCell.H);
            var actualDc = CNTKLib.PastValue(lstmCell.C);

            // Form the recurrence loop by replacing the dh and dc placeholders 
            // with the actualDh and actualDc
            lstmCell.H.ReplacePlaceholders(new Dictionary<Variable, Variable>
            {
                { dh, actualDh },
                { dc, actualDc }
            });

            //return value depending of type of LSTM layer
            if (returnSequence)
                return lstmCell.H;
            else
                return CNTKLib.SequenceLast(lstmCell.H);
        }
    }
}
