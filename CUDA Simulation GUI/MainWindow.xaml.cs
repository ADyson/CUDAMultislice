using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;
using System.Threading;
using Microsoft.Win32;
using GUISimulation;
using BitMiracle.LibTiff.Classic;
using MagnificationClass;

namespace CUDA_Simulation_GUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        

        // Wrapper for unmanaged Code.
        ManagedMultisliceSimulation managedMultislice = new ManagedMultisliceSimulation();

        // List of magnifications for combobox.
        public List<Magnification> MagnificationList { get; set; }
        public List<Resolution> ResolutionList { get; set; }

        Magnification _SelectedScale;
        public Magnification SelectedScale
        {
            get
            {
                return _SelectedScale;
            }
            set
            {
                if (_SelectedScale != value)
                {
                    _SelectedScale = value;
                    RaisePropertyChanged("SelectedScale");
                }
            }
        }

        Resolution _SelectedResolution;
        public Resolution SelectedResolution
        {
            get
            {
                return _SelectedResolution;
            }
            set
            {
                if (_SelectedResolution != value)
                {
                    _SelectedResolution = value;
                    RaisePropertyChanged("SelectedResolution");
                }
            }
        }





        // Event stuff

        public event EventHandler Cancel = delegate { };
        BackgroundWorker worker;

        // File opening dialog
        private static Microsoft.Win32.OpenFileDialog openDialog = new Microsoft.Win32.OpenFileDialog();
        // File saving dialog
        private static Microsoft.Win32.SaveFileDialog saveDialog = new Microsoft.Win32.SaveFileDialog();

        private static WriteableBitmap img;
        public static WriteableBitmap Img
        {
            get { return MainWindow.img; }
            set { MainWindow.img = value; }
        }

        // Values for GUI fields
        int resolutionx;
        int resolutiony;

        float magcalfactor = 1.618f;
        float df;
        float M2f;
        float A2f;
        float M3f;
        float A3f;
        float Cs;
        float Beta;
        float Delta;
        float kV;
        float PixelScale;
        float objectiveAperture;
        double Dose;
        private float slicethickness;

        bool conventional;
        bool GotEW = false;

        public MainWindow()
        {
            InitializeComponent();
            comboBox1.IsEnabled = false;
            radioButton1.IsChecked = true;
            TEMButton.IsChecked = true;
            conventional = true;

            MagnificationList = new List<Magnification>
            {
                new Magnification { Name = "10k", Scale = 4.0f * magcalfactor },
                new Magnification { Name = "25k", Scale = 1.6f * magcalfactor },
                new Magnification { Name = "50k", Scale = 0.8f * magcalfactor },
                new Magnification { Name = "100k", Scale = 0.4f * magcalfactor },
                new Magnification { Name = "150k", Scale = 0.26666f * magcalfactor },
                new Magnification { Name = "200k", Scale = 0.2f * magcalfactor },
                new Magnification { Name = "400k", Scale = 0.1f * magcalfactor },
                new Magnification { Name = "600k", Scale = 0.075f * magcalfactor },
                new Magnification { Name = "800k", Scale = 0.05f * magcalfactor },
                new Magnification { Name = "1M", Scale = 0.04f * magcalfactor },
                new Magnification { Name = "2M", Scale = 0.02f * magcalfactor },
                new Magnification { Name = "4M", Scale = 0.01f * magcalfactor },
                new Magnification { Name = "8M", Scale = 0.005f * magcalfactor },

            };

            ResolutionList = new List<Resolution>
            {
                new Resolution { Name = "4008 x 2672", ResolutionX = 4008, ResolutionY = 2672, Binning = 1 }, // Not sure it can cope properly with this size but also not sure why.
                new Resolution { Name = "2004 x 1336", ResolutionX = 2004, ResolutionY = 1336, Binning = 2 },
                new Resolution { Name = "1002 x 668", ResolutionX = 1002, ResolutionY = 668 , Binning = 4},
                new Resolution { Name = "1024 x 1024", ResolutionX = 1024, ResolutionY = 1024 , Binning = 4},
            };

            DataContext = this;
        }

        void RaisePropertyChanged(string prop)
        {
            if (PropertyChanged != null) { PropertyChanged(this, new PropertyChangedEventArgs(prop)); }
        }
        public event PropertyChangedEventHandler PropertyChanged;

        private void ImportStructureButton(object sender, RoutedEventArgs e)
        {
            // Set defaults for file dialog.
            openDialog.FileName = "file name";                  // Default file name
            openDialog.DefaultExt = ".xyz";                     // Default file extension
            openDialog.Filter = "XYZ Coordinates (.xyz)|*.xyz"; // Filter files by extension

            Nullable<bool> result = openDialog.ShowDialog();

            if (result == true)
            {
                comboBox1.IsEnabled = true;
                // Set label to display filename
                label2.Content = openDialog.FileName;

                // Import from file
                managedMultislice.ImportAtoms(openDialog.FileName);
                managedMultislice.UploadBinnedAtoms();
                managedMultislice.GetParameterisation();
            }
        }

        // Hides some other PreviewTextInput ?
        new private void PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !IsTextAllowedFloatNumber(e.Text);
        }

        private static bool IsTextAllowedFloatNumber(string text)
        {
            Regex regex = new Regex("[^0-9.-]+"); //regex that matches disallowed text
            return !regex.IsMatch(text);
        }

        /// <summary>
        /// TDS CheckBox Checked Function
        /// </summary>
        private void checkBox1_Checked(object sender, RoutedEventArgs e)
        {

        }

        /// <summary>
        /// Resolution Selector
        /// </summary>
        private void comboBox1_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
           // resolution = Convert.ToInt32(comboBox1.SelectedValue.ToString());
            resolutionx = SelectedResolution.ResolutionX;
            resolutiony = SelectedResolution.ResolutionY;
            button4.IsEnabled = true;

        }


        /// <summary>
        /// Simulation Button
        /// </summary>
        private void button3_Click(object sender, RoutedEventArgs e)
        {
            if (TEMButton.IsChecked == true)
            {
                Img = new WriteableBitmap(resolutionx, resolutiony, 96, 96, PixelFormats.Bgr32, null);

                Cancel += CancelProcess;
                progressBar1.Minimum = 0;
                progressBar1.Maximum = 100;

                System.Windows.Threading.Dispatcher mwDispatcher = this.Dispatcher;
                worker = new BackgroundWorker();

                // Changed to alternate model of progress reporting
                //worker.WorkerReportsProgress = true;
                worker.WorkerSupportsCancellation = true;

                float xoff;
                float yoff;
                float.TryParse(textBoxXOff.Text, NumberStyles.Float, null, out xoff);
                float.TryParse(textBoxYOff.Text, NumberStyles.Float, null, out yoff);

                worker.DoWork += delegate(object s, DoWorkEventArgs args)
                {
                    managedMultislice.ApplyMicroscopeParameters(kV, df, M2f, A2f, Cs, Beta, Delta, objectiveAperture);
                    managedMultislice.SetCalculationVariables(SelectedScale.Scale * SelectedResolution.Binning, df, resolutionx, resolutiony, managedMultislice.GetSizeX(), managedMultislice.GetSizeY(), managedMultislice.GetSizeZ(),xoff,yoff);
                    managedMultislice.PreCalculateFrequencies();
                    if (conventional)
                    {
                        managedMultislice.InitialiseWavefunctions(slicethickness);
                    }
                    else
                    {
                        managedMultislice.InitialiseWavefunctions();
                    }

                    //label10.Content = "0 / " + managedMultislice.GetSlices().ToString() + " Slices";

                    for (int i = 1; i <= managedMultislice.GetSlices(conventional); i++)
                    {
                        if (worker.CancellationPending)
                        {
                            args.Cancel = true;
                            return;
                        }
                        if (conventional)
                        {
                            managedMultislice.MultisliceStepConventional(i);
                            // Check after each step to find where it goes wrong
                        }

                        else
                            managedMultislice.MultisliceStepFD(i);

                        // System.Threading.Thread.Sleep(2);

                        if (i % 10 == 0)
                        {
                            UpdateProgressDelegate update = new UpdateProgressDelegate(UpdateProgress);
                            Dispatcher.BeginInvoke(update, i, managedMultislice.GetSlices(conventional));
                        }


                    }
                };

                worker.RunWorkerCompleted += delegate(object s, RunWorkerCompletedEventArgs args)
                {
                    if(GotEW)
                        managedMultislice.FreeExitWave();

                    label10.Content = "Exit Wave Simulated";
                    progressBar1.Value = 100;
                    image1.Source = Img;

                    // Calculate the number of bytes per pixel (should be 4 for this format). 
                    var bytesPerPixel = (Img.Format.BitsPerPixel + 7) / 8;

                    // Stride is bytes per pixel times the number of pixels.
                    // Stride is the byte width of a single rectangle row.
                    var stride = Img.PixelWidth * bytesPerPixel;

                    // Create a byte array for a the entire size of bitmap.
                    var arraySize = stride * Img.PixelHeight;
                    var pixelArray = new byte[arraySize];

                    managedMultislice.GetExitWave();
                    GotEW = true;
                    button12.IsEnabled = true;
                    
      
                    float min = managedMultislice.GetEWMin() - 0.01f;
                    float max = managedMultislice.GetEWMax() + 0.01f;

                    for (int row = 0; row < Img.PixelHeight; row++)
                        for (int col = 0; col < Img.PixelWidth; col++)
                        {
                            pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel] = Convert.ToByte(Math.Ceiling(((managedMultislice.GetEWValueAbs(col, row) - min) / (max - min)) * 254.0f));
                            pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 1] = Convert.ToByte(Math.Ceiling(((managedMultislice.GetEWValueAbs(col, row) - min) / (max - min)) * 254.0f));
                            pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 2] = Convert.ToByte(Math.Ceiling(((managedMultislice.GetEWValueAbs(col, row) - min) / (max - min)) * 254.0f));
                            pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 3] = 0;
                        }

                    //managedMultislice.FreeExitWave();

                    Int32Rect rect = new Int32Rect(0, 0, Img.PixelWidth, Img.PixelHeight);

                    Img.WritePixels(rect, pixelArray, stride, 0);

                    button8.IsEnabled = true;
                    managedMultislice.AllocHostImage();
                };

                worker.RunWorkerAsync();
            }
            else  // THIS IS FOR STEM Simulation
            {
                // Needs to hold 4 images BF/DF/HAADF/Diff
                Img = new WriteableBitmap(resolutionx, resolutiony, 96, 96, PixelFormats.Bgr32, null);

                Cancel += CancelProcess;
                progressBar1.Minimum = 0;
                progressBar1.Maximum = 100;

                System.Windows.Threading.Dispatcher mwDispatcher = this.Dispatcher;
                worker = new BackgroundWorker();

                // Changed to alternate model of progress reporting
                //worker.WorkerReportsProgress = true;
                worker.WorkerSupportsCancellation = true;

                worker.DoWork += delegate(object s, DoWorkEventArgs args)
                {
                    managedMultislice.ApplyMicroscopeParameters(kV, df, M2f, A2f, Cs, Beta, Delta, objectiveAperture);
                    managedMultislice.STEMSetCalculationVariables(SelectedScale.Scale * SelectedResolution.Binning, df,
                        resolutionx, resolutiony, managedMultislice.GetSizeX(), managedMultislice.GetSizeY(),
                        managedMultislice.GetSizeZ());
                    managedMultislice.PreCalculateFrequencies();
                    managedMultislice.AllocTDS();

                    for(int kk = 1; kk <=20; kk++)
                    {
                        managedMultislice.UploadBinnedAtomsTDS();

                        

                        //TODO: loop over all stem positions..
                        int posx = resolutionx/2;
                        int posy = resolutiony/2;

                        managedMultislice.STEMInitialiseWavefunctions(posx, posy);

                        //label10.Content = "0 / " + managedMultislice.GetSlices().ToString() + " Slices";

                        for (int i = 1; i <= managedMultislice.GetSlices(conventional); i++)
                        {
                            if (worker.CancellationPending)
                            {
                                args.Cancel = true;
                                return;
                            }
                            if (conventional)
                            {
                                managedMultislice.MultisliceStepConventional(i);
                                // Check after each step to find where it goes wrong
                            }

                            else
                                managedMultislice.MultisliceStepFD(i);

                            // System.Threading.Thread.Sleep(2);

                            if (i%10 == 0)
                            {
                                UpdateProgressDelegate update = new UpdateProgressDelegate(UpdateProgress);
                                Dispatcher.BeginInvoke(update, i, managedMultislice.GetSlices(conventional));
                            }


                        }

                        // Finished one TDS run, need to store abs of EW somewhere to average....
                        managedMultislice.AddTDSWaves();
                    }
                };

                worker.RunWorkerCompleted += delegate(object s, RunWorkerCompletedEventArgs args)
                {
                   // if (GotEW)
                    //    managedMultislice.FreeExitWave();

                    label10.Content = "Exit Wave Simulated";
                    progressBar1.Value = 100;
                    image1.Source = Img;

                    // Calculate the number of bytes per pixel (should be 4 for this format). 
                    var bytesPerPixel = (Img.Format.BitsPerPixel + 7) / 8;

                    // Stride is bytes per pixel times the number of pixels.
                    // Stride is the byte width of a single rectangle row.
                    var stride = Img.PixelWidth * bytesPerPixel;

                    // Create a byte array for a the entire size of bitmap.
                    var arraySize = stride * Img.PixelHeight;
                    var pixelArray = new byte[arraySize];

                    //managedMultislice.GetSTEMExitWave();
                    GotEW = true;
                    button12.IsEnabled = true;

                    float min = managedMultislice.GetTDSMin() - 0.01f;
                    float max = managedMultislice.GetTDSMax() + 0.01f;

                    for (int row = 0; row < Img.PixelHeight; row++)
                        for (int col = 0; col < Img.PixelWidth; col++)
                        {
                            pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel] = Convert.ToByte(Math.Ceiling(((managedMultislice.GetTDSVal(col, row) - min) / (max - min)) * 254.0f));
                            pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 1] = Convert.ToByte(Math.Ceiling(((managedMultislice.GetTDSVal(col, row) - min) / (max - min)) * 254.0f));
                            pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 2] = Convert.ToByte(Math.Ceiling(((managedMultislice.GetTDSVal(col, row) - min) / (max - min)) * 254.0f));
                            pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 3] = 0;
                        }

                    //managedMultislice.FreeExitWave();

                    Int32Rect rect = new Int32Rect(0, 0, Img.PixelWidth, Img.PixelHeight);

                    Img.WritePixels(rect, pixelArray, stride, 0);

                    button8.IsEnabled = true;
                    managedMultislice.AllocHostImage();
                };

                worker.RunWorkerAsync();
            }


        }

        public delegate void UpdateProgressDelegate(int iteration, int maxIts);

        public void UpdateProgress(int iteration, int maxIts)
        {
            label10.Content = iteration.ToString() + " / " + maxIts.ToString() + " Slices";
            progressBar1.Value = ((float)iteration / (float)maxIts) * 100;
        }

        private void textBox1_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Still isn't sorted completely
            string temp = textBox1.Text;
            Double.TryParse(temp, NumberStyles.Float, null, out Dose);

        }


        private void textBox2_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Still isn't sorted completely
            string temp = textBox2.Text;
            float.TryParse(temp, NumberStyles.Float, null,out df);
         
        }

        private void textBox3_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Still isn't sorted completely
            string temp = textBox3.Text;
            float.TryParse(temp, NumberStyles.Float, null, out kV);
        }

        private void textBox4_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Still isn't sorted completely
            string temp = textBox4.Text;
            float.TryParse(temp, NumberStyles.Float, null, out Beta);
        }

        private void textBox5_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Still isn't sorted completely
            string temp = textBox5.Text;
            float.TryParse(temp, NumberStyles.Float, null, out M2f);
        }

        private void textBox6_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Still isn't sorted completely
            string temp = textBox6.Text;
            float.TryParse(temp, NumberStyles.Float, null, out Cs);
        }

        private void textBox7_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Still isn't sorted completely
            string temp = textBox7.Text;
            float.TryParse(temp, NumberStyles.Float, null, out objectiveAperture);
        }

        private void textBox8_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Still isn't sorted completely
            string temp = textBox8.Text;
            float.TryParse(temp, NumberStyles.Float, null, out Delta);
        }

        private void textBox9_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Still isn't sorted completely
            string temp = textBox9.Text;
            float.TryParse(temp, NumberStyles.Float, null, out A2f);
        }

        private void button2_Click(object sender, RoutedEventArgs e)
        {
            Cancel(sender, e);
        }

        void CancelProcess(object sender, EventArgs e)
        {
            worker.CancelAsync();
        }

        private void button8_Click(object sender, RoutedEventArgs e)
        {
            PixelScale = SelectedScale.Scale * SelectedResolution.Binning;
            //TODO: Add some noise to each pixel
            float dosepix = Convert.ToSingle(Dose * PixelScale * PixelScale);

            managedMultislice.ApplyMicroscopeParameters(kV, df, M2f, A2f, Cs, Beta, Delta, objectiveAperture);
            managedMultislice.SimulateImage(dosepix);


            //TODO: Why only 512,512
            float contrastvalue;
            contrastvalue = managedMultislice.GetImageContrast(0, 0, resolutionx, resolutiony);

            label3.Content = contrastvalue.ToString();
            button6.IsEnabled = true;


        }

        private void button6_Click(object sender, RoutedEventArgs e)
        {
            // Need to get image to calculate noise values before getting DQE'd version of image to form final image.





            // Calculate the number of bytes per pixel (should be 4 for this format). 
            var bytesPerPixel = (Img.Format.BitsPerPixel + 7) / 8;

            // Stride is bytes per pixel times the number of pixels.
            // Stride is the byte width of a single rectangle row.
            var stride = Img.PixelWidth * bytesPerPixel;

            // Create a byte array for a the entire size of bitmap.
            var arraySize = stride * Img.PixelHeight;
            var pixelArray = new byte[arraySize];

            // calculate pixelscale
            float sizx = managedMultislice.GetSizeX();
            float sizy = managedMultislice.GetSizeY();
           // PixelScale = ((sizy > sizx ? 1 : 0) * sizy + (sizx >= sizy ? 1 : 0) * sizx) / resolution;
            PixelScale = SelectedScale.Scale* SelectedResolution.Binning;
            //TODO: Add some noise to each pixel
            double stddev = Math.Sqrt(Dose * PixelScale * PixelScale) / (Dose * PixelScale * PixelScale );
            Random rand = new Random(); //reuse this if you are generating many

            double min = 0.9*managedMultislice.GetImMin()-5;
            double max = 1.1*managedMultislice.GetImMax()+5;

           


            for (int row = 0; row < Img.PixelHeight; row++)
                for (int col = 0; col < Img.PixelWidth; col++)
                {
                    /*
                    double u1 = rand.NextDouble(); //these are uniform(0,1) random doubles
                    double u2 = rand.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                 Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                    double randNormal =
                                 stddev * randStdNormal; //random normal(mean,stdDev^2)
                    if (randNormal < -3 * stddev)
                        randNormal = -3 * stddev;
                    if (randNormal > 3 * stddev)
                        randNormal = 3 * stddev;
                    
                    // Convert.ToByte(Math.Ceiling(((managedMultislice.GetImValue(col, row) - min) / (max - min) + randNormal + 5 * stddev) * 255.0f / (1 + 10 * stddev)))
                    pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel] = Convert.ToByte(Math.Ceiling(((managedMultislice.GetImValue(col, row) + randNormal - min + 3*stddev) / (max - min + 6 * stddev)) * 255.0f ));
                    pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 1] = Convert.ToByte(Math.Ceiling(((managedMultislice.GetImValue(col, row) + randNormal - min + 3 * stddev) / (max - min + 6 * stddev)) * 255.0f));
                    pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 2] = Convert.ToByte(Math.Ceiling(((managedMultislice.GetImValue(col, row) + randNormal - min + 3 * stddev) / (max - min + 6 * stddev)) * 255.0f));
                    pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 3] = 0;
                }
                    */

                    pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel] = Convert.ToByte(Math.Ceiling(((managedMultislice.GetImValue(col, row) - min) / (max - min)) * 254.0f));
                    pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 1] = Convert.ToByte(Math.Ceiling(((managedMultislice.GetImValue(col, row) - min) / (max - min)) * 254.0f));
                    pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 2] = Convert.ToByte(Math.Ceiling(((managedMultislice.GetImValue(col, row) - min) / (max - min)) * 254.0f));
                    pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 3] = 0;
                }
            Int32Rect rect = new Int32Rect(0, 0, Img.PixelWidth, Img.PixelHeight);

            Img.WritePixels(rect, pixelArray, stride, 0);

            button7.IsEnabled = true;
            button6.IsEnabled = false;
            button3.IsEnabled = true;
        }

        private void button7_Click(object sender, RoutedEventArgs e)
        {
            // Calculate the number of bytes per pixel (should be 4 for this format). 
            var bytesPerPixel = (Img.Format.BitsPerPixel + 7) / 8;

            // Stride is bytes per pixel times the number of pixels.
            // Stride is the byte width of a single rectangle row.
            var stride = Img.PixelWidth * bytesPerPixel;

            // Create a byte array for a the entire size of bitmap.
            var arraySize = stride * Img.PixelHeight;
            var pixelArray = new byte[arraySize];

            managedMultislice.GetExitWave();
            float min = 0;
            float max = managedMultislice.GetEWMax();

            for (int row = 0; row < Img.PixelHeight; row++)
                for (int col = 0; col < Img.PixelWidth; col++)
                {
                    pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel] = Convert.ToByte(Math.Ceiling((managedMultislice.GetEWValueAbs(col, row) - min) / (max - min) * 254.0f));
                    pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 1] = Convert.ToByte(Math.Ceiling((managedMultislice.GetEWValueAbs(col, row) - min) / (max - min) * 254.0f));
                    pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 2] = Convert.ToByte(Math.Ceiling((managedMultislice.GetEWValueAbs(col, row) - min) / (max - min) * 254.0f));
                    pixelArray[(row * Img.PixelWidth + col) * bytesPerPixel + 3] = 0;
                }

            managedMultislice.FreeExitWave();

            Int32Rect rect = new Int32Rect(0, 0, Img.PixelWidth, Img.PixelHeight);

            Img.WritePixels(rect, pixelArray, stride, 0);

            button6.IsEnabled = true;
            button7.IsEnabled = false;
        }

        private void radioButton2_Checked(object sender, RoutedEventArgs e)
        {
            radioButton1.IsChecked = false;
            conventional = false;
            textBox2_Copy.IsEnabled = false;
        }

        private void radioButton1_Checked(object sender, RoutedEventArgs e)
        {
            radioButton2.IsChecked = false;
            conventional = true;
            textBox2_Copy.IsEnabled = true;

        }

        private void button3_Click_1(object sender, RoutedEventArgs e)
        {
            saveDialog.Title = "Save Output Image";
            saveDialog.DefaultExt = ".tiff";                     // Default file extension
            saveDialog.Filter = "TIFF Image (.tiff)|*.tiff"; // Filter files by extension

            Nullable<bool> result = saveDialog.ShowDialog();

            if (result == true)
            {
                string filename = saveDialog.FileName;
                using (Tiff output = Tiff.Open(filename, "w"))
                {
                    output.SetField(TiffTag.IMAGEWIDTH, resolutionx);
                    output.SetField(TiffTag.IMAGELENGTH, resolutiony);
                    output.SetField(TiffTag.SAMPLESPERPIXEL, 1);
                    output.SetField(TiffTag.SAMPLEFORMAT, 3);
                    output.SetField(TiffTag.BITSPERSAMPLE, 32);
                    output.SetField(TiffTag.ORIENTATION, BitMiracle.LibTiff.Classic.Orientation.TOPLEFT);
                    output.SetField(TiffTag.ROWSPERSTRIP, resolutiony);
                    output.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
                    output.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
                    output.SetField(TiffTag.COMPRESSION, Compression.NONE);
                    output.SetField(TiffTag.FILLORDER, FillOrder.MSB2LSB);

                    // calculate pixelscale
                    float sizx = managedMultislice.GetSizeX();
                    float sizy = managedMultislice.GetSizeY();
                   // PixelScale = ((sizy > sizx ? 1 : 0) * sizy + (sizx >= sizy ? 1 : 0) * sizx) / resolution;
                    PixelScale = SelectedScale.Scale * SelectedResolution.Binning;

                    //TODO: Add some noise to each pixel
                    double stddev = Math.Sqrt(Dose * PixelScale * PixelScale) / (Dose * PixelScale * PixelScale);
                    Random rand = new Random(); //reuse this if you are generating many


                    for (int i = 0; i < resolutiony; ++i)
                    {
                        float[] buf = new float[resolutionx];
                        byte[] buf2 = new byte[4 * resolutionx];
                        for (int j = 0; j < resolutionx; ++j)
                        {
                            double u1 = rand.NextDouble(); //these are uniform(0,1) random doubles
                            double u2 = rand.NextDouble();
                            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                            double randNormal =
                                         stddev * randStdNormal; //random normal(mean,stdDev^2)

                          //  buf[j] = managedMultislice.GetImValue(j, i) * (1 + Convert.ToSingle(randNormal)); // Why 1 + randomNormal?
                              buf[j] = managedMultislice.GetImValue(j, i);
                        }
                        Buffer.BlockCopy(buf, 0, buf2, 0, buf2.Length);
                        output.WriteScanline(buf2, i);
                    }
                }
            }

        }

        private void button9_Click(object sender, RoutedEventArgs e)
        {
            // Add some validation to ensure i have set simulation parameters before this dialog can be called

            BatchProcessing batchDialog = new BatchProcessing(ref managedMultislice);
            var helper = new WindowInteropHelper(batchDialog);
            var helperthis = new WindowInteropHelper(this); 
            helper.Owner = helperthis.Handle;
            batchDialog.ShowDialog();
        }

        private void RadioButton_Checked_1(object sender, RoutedEventArgs e)
        {
             // STEM Mode - disable image simulation, will all be done via ExitWave button.
            button8.IsEnabled = false;
        }

        private void RadioButton_Checked_2(object sender, RoutedEventArgs e)
        {
            // TEM Mode - Enable Image Simulation again
            button8.IsEnabled = true;
        }

        private void button12_Click(object sender, RoutedEventArgs e)
        {
            if (TEMButton.IsChecked == true)
            {
                saveDialog.Title = "Save Re Exit Wave Image";
                saveDialog.DefaultExt = ".tiff"; // Default file extension
                saveDialog.Filter = "TIFF Image (.tiff)|*.tiff"; // Filter files by extension
                saveDialog.FileName = "EW.tiff";

                Nullable<bool> result = saveDialog.ShowDialog();

                if (result == true)
                {
                    string filename = saveDialog.FileName;
                    using (Tiff output = Tiff.Open(filename, "w"))
                    {
                        output.SetField(TiffTag.IMAGEWIDTH, resolutionx);
                        output.SetField(TiffTag.IMAGELENGTH, resolutiony);
                        output.SetField(TiffTag.SAMPLESPERPIXEL, 1);
                        output.SetField(TiffTag.SAMPLEFORMAT, 3);
                        output.SetField(TiffTag.BITSPERSAMPLE, 32);
                        output.SetField(TiffTag.ORIENTATION, BitMiracle.LibTiff.Classic.Orientation.TOPLEFT);
                        output.SetField(TiffTag.ROWSPERSTRIP, resolutiony);
                        output.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
                        output.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
                        output.SetField(TiffTag.COMPRESSION, Compression.NONE);
                        output.SetField(TiffTag.FILLORDER, FillOrder.MSB2LSB);

                        // calculate pixelscale
                        float sizx = managedMultislice.GetSizeX();
                        float sizy = managedMultislice.GetSizeY();

                        for (int i = 0; i < resolutiony; ++i)
                        {
                            float[] buf = new float[resolutionx];
                            byte[] buf2 = new byte[4*resolutionx];
                            for (int j = 0; j < resolutionx; ++j)
                            {
                                buf[j] = managedMultislice.GetEWValueRe(j, i);
                            }
                            Buffer.BlockCopy(buf, 0, buf2, 0, buf2.Length);
                            output.WriteScanline(buf2, i);
                        }
                    }
                }

                saveDialog.Title = "Save Im Exit Wave Image";
                Nullable<bool> result2 = saveDialog.ShowDialog();
                if (result2 == true)
                {
                    using (Tiff output2 = Tiff.Open(saveDialog.FileName, "w"))
                    {
                        output2.SetField(TiffTag.IMAGEWIDTH, resolutionx);
                        output2.SetField(TiffTag.IMAGELENGTH, resolutiony);
                        output2.SetField(TiffTag.SAMPLESPERPIXEL, 1);
                        output2.SetField(TiffTag.SAMPLEFORMAT, 3);
                        output2.SetField(TiffTag.BITSPERSAMPLE, 32);
                        output2.SetField(TiffTag.ORIENTATION, BitMiracle.LibTiff.Classic.Orientation.TOPLEFT);
                        output2.SetField(TiffTag.ROWSPERSTRIP, resolutiony);
                        output2.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
                        output2.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
                        output2.SetField(TiffTag.COMPRESSION, Compression.NONE);
                        output2.SetField(TiffTag.FILLORDER, FillOrder.MSB2LSB);

                        for (int i = 0; i < resolutiony; ++i)
                        {
                            float[] buf = new float[resolutionx];
                            byte[] buf2 = new byte[4*resolutionx];
                            for (int j = 0; j < resolutionx; ++j)
                            {
                                buf[j] = managedMultislice.GetEWValueIm(j, i);
                            }
                            Buffer.BlockCopy(buf, 0, buf2, 0, buf2.Length);
                            output2.WriteScanline(buf2, i);
                        }
                    }
                }
            }
            else
            {
                saveDialog.Title = "Save CBED";
                saveDialog.DefaultExt = ".tiff"; // Default file extension
                saveDialog.Filter = "TIFF Image (.tiff)|*.tiff"; // Filter files by extension
                saveDialog.FileName = "CBED.tiff";

                Nullable<bool> result = saveDialog.ShowDialog();

                if (result == true)
                {
                    string filename = saveDialog.FileName;
                    using (Tiff output = Tiff.Open(filename, "w"))
                    {
                        output.SetField(TiffTag.IMAGEWIDTH, resolutionx);
                        output.SetField(TiffTag.IMAGELENGTH, resolutiony);
                        output.SetField(TiffTag.SAMPLESPERPIXEL, 1);
                        output.SetField(TiffTag.SAMPLEFORMAT, 3);
                        output.SetField(TiffTag.BITSPERSAMPLE, 32);
                        output.SetField(TiffTag.ORIENTATION, BitMiracle.LibTiff.Classic.Orientation.TOPLEFT);
                        output.SetField(TiffTag.ROWSPERSTRIP, resolutiony);
                        output.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
                        output.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
                        output.SetField(TiffTag.COMPRESSION, Compression.NONE);
                        output.SetField(TiffTag.FILLORDER, FillOrder.MSB2LSB);

                        // calculate pixelscale
                        float sizx = managedMultislice.GetSizeX();
                        float sizy = managedMultislice.GetSizeY();

                        for (int i = 0; i < resolutiony; ++i)
                        {
                            float[] buf = new float[resolutionx];
                            byte[] buf2 = new byte[4 * resolutionx];
                            for (int j = 0; j < resolutionx; ++j)
                            {
                                buf[j] = managedMultislice.GetTDSVal(j, i);
                            }
                            Buffer.BlockCopy(buf, 0, buf2, 0, buf2.Length);
                            output.WriteScanline(buf2, i);
                        }
                    }
                }
            }

        }

        private void comboBox1_mag_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
        }

        private void TextBox2_Copy_OnTextChanged(object sender, TextChangedEventArgs e)
        {
            string temp = textBox2_Copy.Text;
            float.TryParse(temp, NumberStyles.Float, null, out slicethickness);
        }
    }


}
