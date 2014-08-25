using System;
using System.Drawing;
using System.ComponentModel;
using System.Data;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.IO;
using GUISimulation;

namespace CUDA_Simulation_GUI
{
    /// <summary>
    /// Interaction logic for BatchProcessing.xaml
    /// </summary>
    public partial class BatchProcessing : Window
    {
        // Variables for dialog elements.
        float CsStart;
        float CsEnd;
        int CsSteps;
        float dfStart;
        float dfEnd;
        int dfSteps;

        string foldername;

        ManagedMultisliceSimulation managedMultislice;
        private FolderBrowserDialog folderBrowserDialog1;


        public BatchProcessing(ref ManagedMultisliceSimulation managedMultislicein)
        {
            this.managedMultislice = managedMultislicein;
            InitializeComponent();
        }

        // Hides some other PreviewTextInput ?
        new private void PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !IsTextAllowedFloatNumber(e.Text);
        }
        private void PreviewTextInput2(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !IsTextAllowedIntNumber(e.Text);
        }

        private static bool IsTextAllowedFloatNumber(string text)
        {
            Regex regex = new Regex("[^0-9.-]+"); //regex that matches disallowed text
            return !regex.IsMatch(text);
        }

        private static bool IsTextAllowedIntNumber(string text)
        {
            Regex regex = new Regex("^[0-9]+$"); //regex that matches allowed text
            return regex.IsMatch(text);
        }


        

        private void textBox1_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Parse textbox to a float
            string temp = textBox1.Text;
            float.TryParse(temp, NumberStyles.Float, null, out CsStart);
        }

        private void textBox2_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Parse textbox to a float
            string temp = textBox2.Text;
            float.TryParse(temp, NumberStyles.Float, null, out CsEnd);
        }

        private void textBox3_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Parse textbox to a int
            string temp = textBox3.Text;
            int.TryParse(temp, NumberStyles.Integer, null, out CsSteps);
        }

        private void textBox4_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Parse textbox to a float
            string temp = textBox4.Text;
            float.TryParse(temp, NumberStyles.Float, null, out dfStart);
        }

        private void textBox5_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Parse textbox to a float
            string temp = textBox5.Text;
            float.TryParse(temp, NumberStyles.Float, null, out dfEnd);
        }

        private void textBox6_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Parse textbox to a int
            string temp = textBox6.Text;
            int.TryParse(temp, NumberStyles.Integer, null, out dfSteps);
        }

        private void button1_Click(object sender, RoutedEventArgs e)
        {
            folderBrowserDialog1 = new FolderBrowserDialog();
            this.folderBrowserDialog1.RootFolder = System.Environment.SpecialFolder.MyComputer;
            this.folderBrowserDialog1.ShowNewFolderButton = true;

            DialogResult result = this.folderBrowserDialog1.ShowDialog();
            if (result == System.Windows.Forms.DialogResult.OK)
            {
                // retrieve the name of the selected folder
                foldername = this.folderBrowserDialog1.SelectedPath;
            }
        }

        // Start batch processing simulations to output folder
        private void button2_Click(object sender, RoutedEventArgs e)
        {

        }
    }
}
